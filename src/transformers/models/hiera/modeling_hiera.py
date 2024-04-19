# coding=utf-8
# Copyright 2024 Meta and The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch Hiera model."""


import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...modeling_outputs import (
    BackboneOutput,
    BaseModelOutput,
    BaseModelOutputWithPooling,
    ImageClassifierOutput,
    ModelOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from ...utils.backbone_utils import BackboneMixin
from .configuration_hiera import HieraConfig


logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "HieraConfig"

# Base docstring
_CHECKPOINT_FOR_DOC = "EduardoPacheco/hiera-tiny-224"
_EXPECTED_OUTPUT_SHAPE = [1, 49, 768]

# Image classification docstring
_IMAGE_CLASS_CHECKPOINT = "EduardoPacheco/hiera-tiny-224-in1k"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tabby, tabby cat"


HIERA_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "EduardoPacheco/hiera-tiny-224",
    # See all Hiera models at https://huggingface.co/models?filter=hiera
]


@dataclass
class HieraEncoderOutput(ModelOutput):
    """
    Hiera encoder's outputs, with potential hidden states and attentions.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, sequence_length, hidden_size)`. Thesre are the unrolled hidden states of the model.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each stage) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        reshaped_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, height, width, hidden_size)`. These are the reshaped and re-rolled hidden states of the model.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs reshaped to
            include the spatial dimensions.
    """

    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    reshaped_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class HieraModelOutput(ModelOutput):
    """
    Hiera model's outputs that also contains a pooling of the last hidden states.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`, *optional*, returned when `add_pooling_layer=True` is passed):
            Average pooling of the last layer hidden-state.
        mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Tensor indicating which patches are masked (0) and which are not (1).
        ids_restore (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Tensor containing the original index of the (shuffled) masked patches.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, sequence_length, hidden_size)`. These are the unrolled hidden states of the model.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each stage) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        reshaped_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, height, width, hidden_size)`. These are the reshaped and re-rolled hidden states of the model.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs reshaped to
            include the spatial dimensions.
    """

    last_hidden_state: torch.FloatTensor = None
    pooler_output: Optional[torch.FloatTensor] = None
    mask: torch.LongTensor = None
    ids_restore: torch.LongTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    reshaped_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class HieraForImageClassificationOutput(ImageClassifierOutput):
    """
    Hiera image classification outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, `optional`):
            Classification loss.
        logits (`torch.FloatTensor` of shape `(batch_size, num_labels)`):
            Prediction scores of the classification head (logits of the output layer).
        hidden_states (`tuple(torch.FloatTensor)`, `optional`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, sequence_length, hidden_size)`. These are the unrolled hidden states of the model.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, `optional`):
            Tuple of `torch.FloatTensor` (one for each stage) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        reshaped_hidden_states (`tuple(torch.FloatTensor)`, `optional`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, height, width, hidden_size)`. These are the reshaped and re-rolled hidden states of the model.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs reshaped to
            include the spatial dimensions.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    reshaped_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class HieraForPreTrainingOutput(ModelOutput):
    """
    Class for ViTMAEForPreTraining's outputs, with potential hidden states and attentions.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`):
            Pixel reconstruction loss.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, patch_size ** 2 * num_channels)`):
            Pixel reconstruction logits.
        mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Tensor indicating which patches are masked (0) and which are not (1).
        ids_restore (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Tensor containing the original index of the (shuffled) masked patches.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
        reshaped_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, height, width, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs reshaped to include the spatial dimensions.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    mask: torch.LongTensor = None
    ids_restore: torch.LongTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    reshaped_hidden_states: Optional[Tuple[torch.FloatTensor]] = None


# Taken from https://github.com/facebookresearch/hiera/blob/main/hiera/hiera_utils.py#L73
def conv_nd(n: int) -> nn.Module:
    """
    Returns a conv with nd (e.g., Conv2d for n=2). Work up to n=3.
    If you wanted a 4d Hiera, you could probably just implement this for n=4. (no promises)
    """
    return [nn.Identity, nn.Conv1d, nn.Conv2d, nn.Conv3d][n]


# Taken from https://github.com/facebookresearch/hiera/blob/main/hiera/hiera_utils.py#L81
def do_pool(x: torch.Tensor, stride: int) -> torch.Tensor:
    # Refer to `Unroll` to see how this performs a maxpool-Nd
    return x.view(x.shape[0], stride, -1, x.shape[-1]).max(dim=1).values


class HieraPatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config, is_mae: bool = False):
        super().__init__()

        # Support any number of spatial dimensions
        self.spatial_dims = len(config.patch_kernel)
        if self.spatial_dims not in (2, 3):
            raise ValueError(
                f"The number of dimensions of the input image should be 2 or 3, but got {self.spatial_dims}."
            )
        self.num_channels = config.num_channels
        self.image_size = config.input_size[-2:]
        self.tokens_spatial_shape = [i // s for i, s in zip(config.input_size, config.patch_stride)]
        self.mask_spatial_shape = [i // s for i, s in zip(self.tokens_spatial_shape, config.masked_unit_size)]
        self.mask_ratio = config.mask_ratio
        self.is_mae = is_mae

        self.projection = conv_nd(self.spatial_dims)(
            self.num_channels,
            config.embed_dim,
            kernel_size=config.patch_kernel,
            stride=config.patch_stride,
            padding=config.patch_padding,
        )

    def masked_conv(self, pixel_values: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Zero-out the masked regions of the input before conv.
        Prevents leakage of masked regions when using overlapping kernels.
        """
        if mask is None:
            return self.projection(pixel_values)

        target_size = pixel_values.shape[2:]
        # Reshape mask to (batch_size, 1, mask_unit_height, mask_unit_width)
        mask = mask.view(pixel_values.shape[0], 1, *self.mask_spatial_shape)

        if len(mask.shape[2:]) != len(target_size):
            raise ValueError(
                f"The length of the spatial dimensions of the mask should match the one from input image, but got {len(mask.shape[2:])} and {len(target_size)}."
            )

        if mask.shape[2:] != target_size:
            mask = nn.functional.interpolate(mask, size=target_size)

        return self.projection(pixel_values * mask.bool())

    def random_masking(self, pixel_values, noise=None):
        """
        Perform per-sample random masking by per-sample shuffling. Per-sample shuffling is done by argsort random
        noise.

        Args:
            pixel_values (`torch.LongTensor` of shape `(batch_size, num_channels, height, width)`)
            noise (`torch.FloatTensor` of shape `(batch_size, num_mask_units)`, *optional*) which is
                mainly used for testing purposes to control randomness and maintain the reproducibility
        """
        batch_size = pixel_values.shape[0]
        # Tokens selected for masking at mask unit level
        num_windows = math.prod(self.mask_spatial_shape)
        len_keep = int(num_windows * (1 - self.mask_ratio))

        if noise is None:
            noise = torch.rand(batch_size, num_windows, device=pixel_values.device)

        # Sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)
        # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Generate the binary mask: 1 is *keep*, 0 is *remove*
        # Note this is opposite to original MAE
        mask = torch.zeros([batch_size, num_windows], device=pixel_values.device)
        mask[:, :len_keep] = 1
        # Unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return mask, ids_restore

    def forward(
        self,
        pixel_values: torch.Tensor,
        noise: Optional[torch.FloatTensor] = None,
        interpolate_pos_encoding: bool = False,
    ) -> torch.Tensor:
        num_channels = pixel_values.shape[1]
        height, width = pixel_values.shape[-2:]

        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
                f" Expected {self.num_channels} but got {num_channels}."
            )

        if not interpolate_pos_encoding:
            if height != self.image_size[0] or width != self.image_size[1]:
                raise ValueError(
                    f"Input image size ({height}*{width}) doesn't match model"
                    f" ({self.image_size[0]}*{self.image_size[1]})."
                )

        (mask, ids_restore) = self.random_masking(pixel_values, noise=noise) if self.is_mae else (None, None)

        embeddings = self.masked_conv(pixel_values, mask)
        embeddings = embeddings.flatten(2).transpose(2, 1)

        return embeddings, mask, ids_restore


class HieraEmbeddings(nn.Module):
    """
    Construct position and patch embeddings.
    """

    def __init__(self, config: HieraConfig, is_mae: bool = False) -> None:
        super().__init__()
        self.patch_stride = config.patch_stride
        self.tokens_spatial_shape = [i // s for i, s in zip(config.input_size, config.patch_stride)]
        self.mask_spatial_shape = [i // s for i, s in zip(self.tokens_spatial_shape, config.masked_unit_size)]
        self.num_tokens = math.prod(self.tokens_spatial_shape)
        self.sep_pos_embed = config.sep_pos_embed
        self.is_mae = is_mae

        self.patch_embeddings = HieraPatchEmbeddings(config, is_mae=is_mae)

        if self.sep_pos_embed:
            self.position_embeddings_spatial = nn.Parameter(
                torch.zeros(
                    1,
                    self.tokens_spatial_shape[1] * self.tokens_spatial_shape[2],
                    config.embed_dim,
                )
            )
            self.position_embeddings_temporal = nn.Parameter(
                torch.zeros(1, self.tokens_spatial_shape[0], config.embed_dim)
            )
        else:
            self.position_embeddings = nn.Parameter(torch.zeros(1, self.num_tokens, config.embed_dim))

    def interpolate_pos_encoding(
        self, embeddings: torch.Tensor, pos_embeds: torch.Tensor, height: int, width: int
    ) -> torch.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher
        resolution images.

        Adapted from:
        https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174
        """

        num_patches = embeddings.shape[1]
        num_positions = pos_embeds.shape[1]
        if num_patches == num_positions and height == width:
            return pos_embeds
        dim = embeddings.shape[-1]
        h0 = height // self.patch_stride[0] if not self.sep_pos_embed else height // self.patch_stride[1]
        w0 = width // self.patch_stride[1] if not self.sep_pos_embed else width // self.patch_stride[2]
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        h0, w0 = h0 + 0.1, w0 + 0.1
        pos_embeds = pos_embeds.reshape(1, int(math.sqrt(num_positions)), int(math.sqrt(num_positions)), dim)
        pos_embeds = pos_embeds.permute(0, 3, 1, 2)
        pos_embeds = nn.functional.interpolate(
            pos_embeds,
            scale_factor=(h0 / math.sqrt(num_positions), w0 / math.sqrt(num_positions)),
            mode="bicubic",
            align_corners=False,
        )
        if int(h0) != pos_embeds.shape[-2] or int(w0) != pos_embeds.shape[-1]:
            raise ValueError("The interpolated position encoding does not have the right size")
        pos_embeds = pos_embeds.permute(0, 2, 3, 1).view(1, -1, dim)
        return pos_embeds

    def get_position_embedding(
        self, embeddings: torch.Tensor, height: int, width: int, interpolate_pos_encoding: bool
    ) -> torch.Tensor:
        if self.sep_pos_embed:
            spatial = self.position_embeddings_spatial
            spatial = (
                self.interpolate_pos_encoding(embeddings, spatial, height, width)
                if interpolate_pos_encoding
                else spatial
            )
            spatial = spatial.repeat(1, self.tokens_spatial_shape[0], 1)

            temporal = torch.repeat_interleave(
                self.position_embeddings_temporal,
                self.tokens_spatial_shape[1] * self.tokens_spatial_shape[2],
                dim=1,
            )

            return spatial + temporal
        else:
            position_embeddings = self.position_embeddings
            position_embeddings = (
                self.interpolate_pos_encoding(embeddings, position_embeddings, height, width)
                if interpolate_pos_encoding
                else position_embeddings
            )
            return position_embeddings

    def forward(
        self,
        pixel_values: torch.Tensor,
        noise: Optional[torch.FloatTensor] = None,
        interpolate_pos_encoding: bool = False,
    ) -> torch.Tensor:
        if len(self.tokens_spatial_shape) == 2:
            batch_size, num_channels, height, width = pixel_values.shape
        else:
            batch_size, num_channels, depth, height, width = pixel_values.shape

        embeddings, mask, ids_restore = self.patch_embeddings(
            pixel_values, noise=noise, interpolate_pos_encoding=interpolate_pos_encoding
        )

        embeddings = embeddings + self.get_position_embedding(embeddings, height, width, interpolate_pos_encoding)

        return embeddings, mask, ids_restore


class HieraMaskUnitAttention(nn.Module):
    """
    Computes either Mask Unit or Global Attention. Also is able to perform q pooling.

    Note: this assumes the tokens have already been flattened and unrolled into mask units.
    """

    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        query_stride: int = 1,
        window_size: int = 0,
        use_mask_unit_attn: bool = False,
    ):
        super().__init__()

        self.dim = dim
        self.dim_out = dim_out
        self.num_heads = num_heads
        self.query_stride = query_stride

        self.head_dim = dim_out // num_heads
        self.scale = (self.head_dim) ** -0.5

        self.qkv = nn.Linear(dim, 3 * dim_out)
        self.proj = nn.Linear(dim_out, dim_out)

        self.window_size = window_size
        self.use_mask_unit_attn = use_mask_unit_attn

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: bool = False,
    ) -> torch.Tensor:
        """Input should be of shape [batch, tokens, channels]."""
        batch_size, seq_len, _ = hidden_states.shape

        num_windows = 1
        if self.use_mask_unit_attn:
            num_windows = seq_len // (self.query_stride * self.window_size)

        qkv = self.qkv(hidden_states)
        qkv = qkv.reshape(batch_size, -1, num_windows, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(3, 0, 4, 2, 1, 5)

        query, key, value = qkv.unbind(0)

        if self.query_stride > 1:
            # Refer to Unroll to see how this performs a maxpool-Nd
            query = query.view(batch_size, self.num_heads, num_windows, self.query_stride, -1, self.head_dim)
            query = query.max(dim=3).values

        attn_weights = (query * self.scale) @ key.transpose(-1, -2)
        attn_weights = attn_weights.softmax(dim=-1)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = attn_weights @ value
        attn_output = attn_output.transpose(1, 3).reshape(batch_size, -1, self.dim_out)
        attn_output = self.proj(attn_output)

        return (attn_output, attn_weights) if output_attentions else (attn_output, None)


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


# Copied from transformers.models.beit.modeling_beit.BeitDropPath with Beit->Hiera
class HieraDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class HieraMlp(nn.Module):
    def __init__(self, config, dim: int):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(dim, int(dim * config.mlp_ratio))
        self.fc2 = nn.Linear(int(dim * config.mlp_ratio), dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class HieraLayer(nn.Module):
    def __init__(
        self,
        config,
        dim: int,
        dim_out: int,
        num_heads: int,
        drop_path: float = 0.0,
        query_stride: int = 1,
        window_size: int = 0,
        use_mask_unit_attn: bool = False,
    ):
        super().__init__()

        self.dim = dim
        self.dim_out = dim_out
        self.query_stride = query_stride

        self.layernorm_before = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        self.attn = HieraMaskUnitAttention(dim, dim_out, num_heads, query_stride, window_size, use_mask_unit_attn)

        self.layernorm_after = nn.LayerNorm(dim_out, eps=config.layer_norm_eps)
        self.mlp = HieraMlp(config, dim_out)

        self.drop_path = HieraDropPath(drop_path) if drop_path > 0 else nn.Identity()
        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: bool = False,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        # Attention + Q Pooling
        hidden_states_norm = self.layernorm_before(hidden_states)
        if self.dim != self.dim_out:
            hidden_states = self.proj(hidden_states_norm)
            # Refer to `HieraUnroll` to see how this performs a maxpool-Nd
            hidden_states = hidden_states.view(batch_size, self.query_stride, -1, self.dim_out).max(dim=1).values

        (hidden_states_norm, attn_weights) = self.attn(
            hidden_states_norm, head_mask, output_attentions=output_attentions
        )
        hidden_states = hidden_states + self.drop_path(hidden_states_norm)

        residual = hidden_states
        hidden_states = self.layernorm_after(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.drop_path(hidden_states)

        return (hidden_states, attn_weights)


class HieraStage(nn.Module):
    def __init__(
        self,
        config,
        depth: int,
        dim: int,
        dim_out: int,
        num_heads: int,
        drop_path: List[float],
        query_stride: List[int],
        window_size: int,
        use_mask_unit_attn: bool,
        stage_num: Optional[int] = None,
    ) -> None:
        super().__init__()
        # we need to know if the previous stage used masked attention
        # mask unit or global attention.
        # lag by 1 layer, so that global attention,
        # applied post pooling on lower resolution
        previous_stage_used_masked_attention = False
        if stage_num is not None:
            previous_stage_used_masked_attention = config.masked_unit_attention[stage_num - 1 if stage_num > 0 else 0]
        self.layers = nn.ModuleList(
            [
                HieraLayer(
                    config=config,
                    dim=dim if i == 0 else dim_out,
                    dim_out=dim_out,
                    num_heads=num_heads,
                    drop_path=drop_path[i],
                    query_stride=query_stride[i],
                    window_size=window_size,
                    use_mask_unit_attn=use_mask_unit_attn or (previous_stage_used_masked_attention and i == 0),
                )
                for i in range(depth)
            ]
        )

    def forward(
        self, hidden_states: torch.Tensor, head_mask: Optional[torch.FloatTensor], output_attentions: bool = False
    ) -> torch.Tensor:
        for i, layer_module in enumerate(self.layers):
            layer_head_mask = head_mask[i] if head_mask is not None else None
            (hidden_states, attn_weights) = layer_module(
                hidden_states, layer_head_mask, output_attentions=output_attentions
            )

        return hidden_states, attn_weights


def undo_windowing(hidden_states: torch.Tensor, shape: List[int], mask_unit_shape: List[int]) -> torch.Tensor:
    """
    Restore spatial organization by undoing windowed organization of mask units.
    """
    num_dims = len(shape)
    batch_size, hidden_size = hidden_states.shape[0], hidden_states.shape[-1]
    # From: [batch_size, num_mask_unit_height*num_#mask_unit_wdith, mask_unit_height, mask_unit_width, hidden_size]
    # To: [batch_size, num_mask_unit_height, num_mask_unit_width, mask_unit_height, mask_unit_width, hidden_size]
    num_mask_units = [s // mu for s, mu in zip(shape, mask_unit_shape)]
    hidden_states = hidden_states.view(batch_size, *num_mask_units, *mask_unit_shape, hidden_size)

    # From: [batch_size, num_mask_unit_height, num_mask_unit_width, mask_unit_height, mask_unit_width, hidden_size]
    # To: [batch_size, num_mask_unit_height*mask_unit_height, num_mask_unit_width*mask_unit_width, hidden_size]
    permute = (
        [0]
        + sum(
            [list(p) for p in zip(range(1, 1 + num_dims), range(1 + num_dims, 1 + 2 * num_dims))],
            [],
        )
        + [len(hidden_states.shape) - 1]
    )
    hidden_states = hidden_states.permute(permute).reshape(batch_size, *shape, hidden_size)

    return hidden_states


class HieraEncoder(nn.Module):
    def __init__(self, config: HieraConfig) -> None:
        super().__init__()
        self.config = config

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(config.depths))]
        # query strides rule
        stage_ends = [sum(config.depths[:i]) - 1 for i in range(1, len(config.depths) + 1)]
        query_pool_layer = [stage_end + 1 for stage_end in stage_ends[: config.num_query_pool]]
        query_strides = [
            math.prod(config.query_stride) if i in query_pool_layer else 1 for i in range(sum(config.depths))
        ]

        # Transformer blocks
        self.stages = nn.ModuleList()
        embed_dim = config.embed_dim

        for idx_stage, depth in enumerate(config.depths):
            dim_out = int(config.embed_dim * config.embed_dim_multiplier**idx_stage)

            stage = HieraStage(
                config=config,
                depth=depth,
                dim=embed_dim,
                dim_out=dim_out,
                num_heads=int(config.initial_num_heads * config.num_head_multiplier**idx_stage),
                drop_path=dpr[sum(config.depths[:idx_stage]) : sum(config.depths[: idx_stage + 1])],
                query_stride=query_strides[sum(config.depths[:idx_stage]) : sum(config.depths[: idx_stage + 1])],
                window_size=int(math.prod(config.masked_unit_size) * math.prod(config.query_stride) ** -idx_stage),
                use_mask_unit_attn=config.masked_unit_attention[idx_stage],
                stage_num=idx_stage,
            )

            embed_dim = dim_out
            self.stages.append(stage)

        # Setting reroll schedule
        # The first stage has to reverse everything
        # The next stage has to reverse all but the first unroll, etc.
        stage_size = [i // s for i, s in zip(config.input_size, config.patch_stride)]
        unroll_schedule = [config.query_stride] * len(config.depths[:-1])

        self.schedule = {}
        for idx_stage in range(len(config.depths)):
            self.schedule[idx_stage] = unroll_schedule, stage_size
            if idx_stage < config.num_query_pool:
                stage_size = [i // s for i, s in zip(stage_size, config.query_stride)]
                unroll_schedule = unroll_schedule[1:]

        self.gradient_checkpointing = False

    def reroll(
        self, hidden_states: torch.Tensor, stage_idx: int, mask: Optional[torch.BoolTensor] = None
    ) -> torch.Tensor:
        """
        Roll the given tensor back up to spatial order assuming it's from the given block.

        If no mask is provided returns:
            - [batch_size, height, width, hidden_size] for 2d
            - [batch_size, frames, height, width, hidden_size] for 3d
        If a mask is provided returns:
            - [batch_size, num_mask_units, mask_unit_height, mask_unit_width, hidden_size] for 2d
        """
        schedule, size = self.schedule[stage_idx]
        batch_size, seq_len, hidden_size = hidden_states.shape

        num_dim = len(size)
        mask_unit_shape = [1] * num_dim

        for strides in schedule:
            # Extract the current patch from seq_len
            hidden_states = hidden_states.view(
                batch_size, *strides, seq_len // math.prod(strides), *mask_unit_shape, hidden_size
            )

            # Move that patch into the current MU
            # Example in 2d:
            # Input: [batch_size, stride, stride, seq_len//(stride*stride), mask_unit_height, mask_unit_width, hidden_size]
            # Output: [batch_size, seq_len//(stride*stride), stride, mask_unit_height, stride, mask_unit_width, hidden_size]
            L = len(hidden_states.shape)
            permute = (
                [0, 1 + num_dim]
                + sum(
                    [list(p) for p in zip(range(1, 1 + num_dim), range(1 + num_dim + 1, L - 1))],
                    [],
                )
                + [L - 1]
            )
            hidden_states = hidden_states.permute(permute)

            # Reshape to [batch_size, seq_len//(stride*stride), *mask_units, hidden_size]
            for i in range(num_dim):
                mask_unit_shape[i] *= strides[i]
            hidden_states = hidden_states.reshape(batch_size, -1, *mask_unit_shape, hidden_size)
            seq_len = hidden_states.shape[1]

        # Current shape (e.g., 2d: [batch_size, #num_mask_units_height*#num_mask_units_width, mask_unit_height, mask_unit_width, hidden_size])
        hidden_states = hidden_states.view(batch_size, seq_len, *mask_unit_shape, hidden_size)

        # If masked, return [batch_size, num_mask_units, mask_unit_height, mask_unit_width, hidden_size]
        if mask is not None:
            return hidden_states

        # If not masked, we can return [batch_size, height, width, hidden_size]
        hidden_states = undo_windowing(hidden_states, size, mask_unit_shape)

        return hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_reshaped_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
            reshaped_hidden_states = self.reroll(hidden_states, stage_idx=0, mask=mask)
            all_reshaped_hidden_states = all_reshaped_hidden_states + (reshaped_hidden_states,)

        for i, stage_module in enumerate(self.stages):
            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    stage_module.__call__, hidden_states, layer_head_mask, output_attentions
                )
            else:
                layer_outputs = stage_module(hidden_states, layer_head_mask, output_attentions)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
                reshaped_hidden_states = self.reroll(hidden_states, stage_idx=i, mask=mask)
                all_reshaped_hidden_states = all_reshaped_hidden_states + (reshaped_hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return HieraEncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            reshaped_hidden_states=all_reshaped_hidden_states,
        )


def unroll(hidden_states: torch.Tensor, size: List[int], schedule: List[List[int]]) -> torch.Tensor:
    """
    Reorders the tokens such that patches are contiguous in memory.
    E.g., given [batch_size, (height, width), hidden_size] and stride of (stride, stride), this will re-order the tokens as
    [batch_size, (stride, stride, height // stride, width // stride), hidden_size]

    This allows operations like Max2d to be computed as x.view(batch_size, stride*stride, -1, hidden_size).max(dim=1).
    Not only is this faster, but it also makes it easy to support inputs of arbitrary
    dimensions in addition to patch-wise sparsity.

    Performing this operation multiple times in sequence puts entire windows as contiguous
    in memory. For instance, if you applied the stride (2, 2) 3 times, entire windows of
    size 8x8 would be contiguous in memory, allowing operations like mask unit attention
    computed easily and efficiently, while also allowing max to be applied sequentially.

    Note: This means that intermediate values of the model are not in height x width order, so they
    need to be re-rolled if you want to use the intermediate values as a height x width feature map.
    The last block of the network is fine though, since by then the strides are all consumed.
    """
    batch_size, _, hidden_size = hidden_states.shape

    current_size = size
    hidden_states = hidden_states.view(*([batch_size] + current_size + [hidden_size]))

    for strides in schedule:
        # Move patches with the given strides to the batch dimension

        # Create a view of the tensor with the patch stride as separate dims
        # For example in 2d: [batch_size, height // stride, stride, width // stride, stride, C]
        current_size = [i // s for i, s in zip(current_size, strides)]
        # initialize new_shape with [height // stride, stride, width // stride, stride]
        new_shape = [item for pair in zip(current_size, strides) for item in pair]
        # add batch_size and hidden_size to new_shape
        new_shape = [batch_size] + new_shape + [hidden_size]
        hidden_states = hidden_states.view(new_shape)

        # Move the patch stride into the batch dimension
        # For example in 2d: [batch_size, stride, stride, height // stride, width // stride, hidden_size]
        num_dims = len(new_shape)
        permute = [0] + list(range(2, num_dims - 1, 2)) + list(range(1, num_dims - 1, 2)) + [num_dims - 1]
        hidden_states = hidden_states.permute(permute)

        # Now finally flatten the relevant dims into the batch dimension
        hidden_states = hidden_states.flatten(0, len(strides))
        batch_size *= math.prod(strides)

    hidden_states = hidden_states.reshape(-1, math.prod(size), hidden_size)
    return hidden_states


class HieraPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = HieraConfig
    base_model_prefix = "hiera"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True

    def _init_weights(self, module) -> None:
        """Initialize the weights"""
        std = self.config.initializer_range

        if isinstance(module, HieraEmbeddings):
            if self.config.sep_pos_embed:
                nn.init.trunc_normal_(module.position_embeddings_spatial, std=std)
                nn.init.trunc_normal_(module.position_embeddings_temporal, std=std)
            else:
                nn.init.trunc_normal_(module.position_embeddings, std=std)

        elif isinstance(module, HieraDecoder):
            nn.init.trunc_normal_(module.mask_token, std=std)
            nn.init.trunc_normal_(module.decoder_position_embeddings, std=std)

        elif isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            nn.init.trunc_normal_(module.weight, std=std)
            if module.bias is not None:
                nn.init.constant_(module.bias, std)

        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, std)
            nn.init.constant_(module.weight, self.config.layer_norm_init)


HIERA_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`HieraConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

HIERA_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See [`BitImageProcessor.__call__`]
            for details.

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
        interpolate_pos_encoding (`bool`, *optional*):
            Whether to interpolate the pre-trained position encodings.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


class HieraPooler(nn.Module):
    def __init__(self, config: HieraConfig):
        super().__init__()
        num_features = int(config.embed_dim * config.embed_dim_multiplier ** (len(config.depths) - 1))
        self.layernorm = nn.LayerNorm(num_features, eps=config.layer_norm_eps)
        self.pooler = nn.AdaptiveAvgPool1d(1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.transpose(1, 2)
        pooled_output = self.pooler(hidden_states)
        pooled_output = torch.flatten(pooled_output, 1)
        pooled_output = self.layernorm(pooled_output)
        return pooled_output


@add_start_docstrings(
    "The bare Hiera Model transformer outputting raw hidden-states without any specific head on top.",
    HIERA_START_DOCSTRING,
    """
        add_pooling_layer (`bool`, *optional*, defaults to `True`):
                Whether or not to apply pooling layer.
        is_mae (`bool`, *optional*, defaults to `False`):
                Whether or not to run the model on MAE mode.
    """,
)
class HieraModel(HieraPreTrainedModel):
    def __init__(self, config: HieraConfig, add_pooling_layer: bool = True, is_mae: bool = False):
        super().__init__(config)
        self.num_features = int(config.embed_dim * config.embed_dim_multiplier ** (len(config.depths) - 1))

        self.embeddings = HieraEmbeddings(config, is_mae=is_mae)
        self.encoder = HieraEncoder(config)

        self.unroll_size = [i // s for i, s in zip(config.input_size, config.patch_stride)]
        self.unroll_schedule = [config.query_stride] * len(config.depths[:-1])

        self.pooler = HieraPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> HieraPatchEmbeddings:
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune: Dict[int, List[int]]) -> None:
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(HIERA_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=HieraModelOutput,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        noise: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        noise (`torch.FloatTensor` of shape `(batch_size, num_mask_units)`, *optional*) which is
                mainly used for testing purposes to control randomness and maintain the reproducibility
                when is_mae is set to True.
        """
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

        # TODO: maybe have a cleaner way to cast the input (from `ImageProcessor` side?)
        expected_dtype = self.embeddings.patch_embeddings.projection.weight.dtype
        if pixel_values.dtype != expected_dtype:
            pixel_values = pixel_values.to(expected_dtype)

        embedding_output, mask, ids_restore = self.embeddings(
            pixel_values, interpolate_pos_encoding=interpolate_pos_encoding, noise=noise
        )

        hidden_states = unroll(embedding_output, self.unroll_size, self.unroll_schedule)

        # Discard masked tokens if mask is provided
        if mask is not None:
            mask_unit_area = math.prod(self.config.masked_unit_size)
            batch_size, _, hidden_size = hidden_states.shape
            positions = mask.unsqueeze(-1).tile(1, mask_unit_area, hidden_size)
            positions = positions.bool()
            hidden_states = hidden_states[positions]
            hidden_states = hidden_states.view(batch_size, -1, hidden_size)

        encoder_outputs = self.encoder(
            hidden_states,
            mask=mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = None
        if self.pooler is not None:
            pooled_output = self.pooler(sequence_output)

        if not return_dict:
            head_outputs = (sequence_output, pooled_output) if pooled_output is not None else (sequence_output,)
            head_outputs = head_outputs + (mask, ids_restore) if mask is not None else head_outputs
            return head_outputs + encoder_outputs[1:]

        return HieraModelOutput(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            mask=mask,
            ids_restore=ids_restore,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            reshaped_hidden_states=encoder_outputs.reshaped_hidden_states,
        )


class HieraDecoder(nn.Module):
    def __init__(self, config: HieraConfig):
        super().__init__()
        num_features = int(config.embed_dim * config.embed_dim_multiplier ** (len(config.depths) - 1))
        self.tokens_spatial_shape = [i // s for i, s in zip(config.input_size, config.patch_stride)]
        self.tokens_spatial_shape_final = [
            i // s ** (config.num_query_pool) for i, s in zip(self.tokens_spatial_shape, config.query_stride)
        ]
        self.mask_unit_spatial_shape_final = [
            i // s ** (config.num_query_pool) for i, s in zip(config.masked_unit_size, config.query_stride)
        ]

        self.decoder_embeddings = nn.Linear(num_features, config.decoder_embed_dim)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.decoder_embed_dim))

        self.decoder_position_embeddings = nn.Parameter(
            torch.zeros(1, math.prod(self.tokens_spatial_shape_final), config.decoder_embed_dim)
        )

        self.decoder_block = HieraStage(
            config=config,
            dim=config.decoder_embed_dim,
            dim_out=config.decoder_embed_dim,
            num_heads=config.decoder_num_heads,
            depth=config.decoder_depth,
            use_mask_unit_attn=False,
            drop_path=[0.0] * config.decoder_depth,
            query_stride=[1] * config.decoder_depth,
            window_size=0,
        )

        self.decoder_norm = nn.LayerNorm(config.decoder_embed_dim, eps=config.layer_norm_eps)

        # patch stride of prediction
        self.pred_stride = config.patch_stride[-1] * (config.query_stride[-1] ** config.num_query_pool)
        pred_dim = (self.pred_stride ** len(config.query_stride)) * config.num_channels

        self.decoder_pred = nn.Linear(config.decoder_embed_dim, pred_dim)

    def forward(
        self,
        encoder_hidden_states: torch.Tensor,
        mask: torch.BoolTensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> torch.Tensor:
        # Embed tokens
        hidden_states = self.decoder_embeddings(encoder_hidden_states)

        # Combine visible and mask tokens

        # hidden_states : [batch_size, num_mask_units_visible, *mask_unit_spatial_shape_final, decoder_embed_dim]
        # mask: [batch_size, num_mask_units]
        decoder_hidden_states = torch.zeros(
            *mask.shape, *hidden_states.shape[2:], device=hidden_states.device, dtype=hidden_states.dtype
        )
        mask_tokens = self.mask_token.view((1,) * (len(mask.shape) + len(hidden_states.shape[2:-1])) + (-1,))
        new_mask_shape = mask.shape + (1,) * len(hidden_states.shape[2:])
        mask = mask.reshape(new_mask_shape)
        expand_shape = (-1,) * 2 + hidden_states.shape[2:]
        mask = mask.expand(expand_shape)
        decoder_hidden_states[mask.bool()] = hidden_states.flatten()
        decoder_hidden_states = (1 - mask) * mask_tokens + mask * decoder_hidden_states

        # Get back spatial order
        hidden_states = undo_windowing(
            decoder_hidden_states,
            self.tokens_spatial_shape_final,
            self.mask_unit_spatial_shape_final,
        )
        mask = undo_windowing(
            mask[..., 0:1],
            self.tokens_spatial_shape_final,
            self.mask_unit_spatial_shape_final,
        )

        # Flatten
        hidden_states = hidden_states.reshape(hidden_states.shape[0], -1, hidden_states.shape[-1])
        mask = mask.view(hidden_states.shape[0], -1)

        # Add pos embed
        hidden_states = hidden_states + self.decoder_position_embeddings

        # Apply decoder blocks
        hidden_states, attn_weights = self.decoder_block(
            hidden_states, head_mask=head_mask, output_attentions=output_attentions
        )
        hidden_states = self.decoder_norm(hidden_states)

        # Predictor projection
        hidden_states = self.decoder_pred(hidden_states)

        return hidden_states, mask


class HieraMultiScaleHead(nn.Module):
    def __init__(self, config: HieraConfig):
        super().__init__()
        self.mask_unit_spatial_shape_final = [
            i // s ** (config.num_query_pool) for i, s in zip(config.masked_unit_size, config.query_stride)
        ]
        self.stage_dimensions = [
            int(config.embed_dim * config.embed_dim_multiplier**i) for i in range(len(config.depths))
        ]
        current_masked_unit_size = config.masked_unit_size
        self.multi_scale_fusion_heads = nn.ModuleList()

        for idx in range(config.num_query_pool):
            kernel = [i // s for i, s in zip(current_masked_unit_size, self.mask_unit_spatial_shape_final)]
            current_masked_unit_size = [i // s for i, s in zip(current_masked_unit_size, config.query_stride)]
            self.multi_scale_fusion_heads.append(
                conv_nd(len(config.query_stride))(
                    self.stage_dimensions[idx],
                    self.stage_dimensions[-1],
                    kernel_size=kernel,
                    stride=kernel,
                )
            )
        self.multi_scale_fusion_heads.append(nn.Identity())

    def apply_fusion_head(self, head: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
        if isinstance(head, nn.Identity):
            return hidden_states

        batch_size, num_mask_units = hidden_states.shape[0:2]
        # From: [batch_size, num_mask_units, mask_unit_height, mask_unit_width, hidden_size]
        # To: head([batch_size * num_mask_units, hidden_size, mask_unit_height, mask_unit_width])
        permute = [0] + [len(hidden_states.shape) - 2] + list(range(1, len(hidden_states.shape) - 2))
        hidden_states = hidden_states.reshape(batch_size * num_mask_units, *hidden_states.shape[2:])
        hidden_states = hidden_states.permute(permute)
        hidden_states = head(hidden_states)

        # Restore original layout
        permute = [0] + list(range(2, len(hidden_states.shape))) + [1]
        hidden_states = hidden_states.permute(permute)
        hidden_states = hidden_states.reshape(
            batch_size, num_mask_units, *hidden_states.shape[1:-1], hidden_states.shape[-1]
        )
        return hidden_states

    def forward(self, feature_maps: List[torch.Tensor]) -> torch.Tensor:
        # Multi-scale fusion
        hidden_states = 0.0
        for head, feature_map in zip(self.multi_scale_fusion_heads, feature_maps):
            hidden_states = hidden_states + self.apply_fusion_head(head, feature_map)

        return hidden_states


@add_start_docstrings(
    """The Hiera Model transformer with the decoder on top for self-supervised pre-training.

    <Tip>

    Note that we provide a script to pre-train this model on custom data in our [examples
    directory](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-pretraining).

    </Tip>
    """,
    HIERA_START_DOCSTRING,
)
class HieraForPreTraining(HieraPreTrainedModel):
    def __init__(self, config: HieraConfig) -> None:
        super().__init__(config)
        # Encoder
        self.hiera = HieraModel(config, add_pooling_layer=False, is_mae=True)
        self.encoder_norm = nn.LayerNorm(self.hiera.num_features, eps=config.layer_norm_eps)
        # Multi-scale fusion heads
        self.multiscale_fusion = HieraMultiScaleHead(config)
        # Decoder
        self.decoder = HieraDecoder(config)
        self.pred_stride = self.decoder.pred_stride

        # Initialize weights and apply final processing
        self.post_init()

    def get_pixel_label_2d(self, pixel_values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # mask (boolean tensor): True means *masked*
        pixel_values = pixel_values.permute(0, 2, 3, 1)

        size = self.pred_stride
        label = pixel_values.unfold(1, size, size).unfold(2, size, size)
        label = label.flatten(1, 2).flatten(2)
        label = label[mask.bool()]
        if self.config.norm_pix_loss:
            mean = label.mean(dim=-1, keepdim=True)
            var = label.var(dim=-1, keepdim=True)
            label = (label - mean) / (var + 1.0e-6) ** 0.5

        return label

    def get_pixel_label_3d(self, pixel_values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # mask (boolean tensor): True means *masked*
        pixel_values = pixel_values[:, :, :: self.patch_stride[0], :, :]

        size = self.pred_stride
        label = pixel_values.unfold(3, size, size).unfold(4, size, size)
        # Different from 2D
        label = label.permute(0, 2, 3, 4, 5, 6, 1)
        label = label.flatten(1, 3).flatten(2)
        label = label[mask.bool()]
        if self.config.norm_pix_loss:
            mean = label.mean(dim=-1, keepdim=True)
            var = label.var(dim=-1, keepdim=True)
            label = (label - mean) / (var + 1.0e-6) ** 0.5

        return label

    def forward_loss(self, pixel_values: torch.Tensor, logits: torch.Tensor, mask: torch.BoolTensor):
        # We invert the mask such that 1.0 is *masked*
        mask = 1 - mask
        if len(self.config.query_stride) == 2:
            label = self.get_pixel_label_2d(pixel_values, mask)
        elif len(self.config.query_stride) == 3:
            label = self.get_pixel_label_3d(pixel_values, mask)
        else:
            raise NotImplementedError("Only images and videos are supported")

        logits = logits[mask.bool()]
        loss = (logits - label) ** 2
        loss = loss.mean()

        return loss

    @add_start_docstrings_to_model_forward(HIERA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=HieraForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        noise: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, HieraForPreTrainingOutput]:
        r"""
        noise (`torch.FloatTensor` of shape `(batch_size, num_mask_units)`, *optional*) which is
                mainly used for testing purposes to control randomness and maintain the reproducibility
                when is_mae is set to True.

        Returns:

        Examples:
        ```python
        >>> from transformers import AutoImageProcessor, HieraForPreTraining
        >>> import torch
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("EduardoPacheco/hiera-tiny-224-mae")
        >>> model = HieraForPreTraining.from_pretrained("EduardoPacheco/hiera-tiny-224-mae")

        >>> inputs = image_processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> logits = outputs.logits
        >>> list(logits.shape)
        [1, 196, 768]
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        outputs = self.hiera(
            pixel_values,
            noise=noise,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=True,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=True,
        )

        feature_maps = outputs.reshaped_hidden_states
        mask = outputs.mask
        ids_to_restore = outputs.ids_restore
        # Take only the query pooled and last hidden states
        feature_maps = feature_maps[1 : self.hiera.config.num_query_pool + 1] + (feature_maps[-1],)
        fused_hidden_states = self.multiscale_fusion(feature_maps)
        fused_hidden_states = self.encoder_norm(fused_hidden_states)

        # Reconstruct pixel values
        logits, mask = self.decoder(
            fused_hidden_states,
            mask=mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
        )

        loss = self.forward_loss(pixel_values, logits, mask)

        if not return_dict:
            output = (logits, mask, ids_to_restore)
            if output_hidden_states:
                output = output + (outputs.hidden_states,)
            if output_attentions:
                output = output + (outputs.attentions,)
            if output_hidden_states:
                output = output + (outputs.reshaped_hidden_states,)
            return ((loss,) + output) if loss is not None else output

        return HieraForPreTrainingOutput(
            loss=loss,
            logits=logits,
            mask=mask,
            ids_restore=ids_to_restore,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions,
            reshaped_hidden_states=outputs.reshaped_hidden_states if output_hidden_states else None,
        )


@add_start_docstrings(
    """
    Hiera Model transformer with an image classification head on top (a linear layer on top of the final hidden state with
    average pooling) e.g. for ImageNet.

    <Tip>

        Note that it's possible to fine-tune Hiera on higher resolution images than the ones it has been trained on, by
        setting `interpolate_pos_encoding` to `True` in the forward of the model. This will interpolate the pre-trained
        position embeddings to the higher resolution.

    </Tip>
    """,
    HIERA_START_DOCSTRING,
)
class HieraForImageClassification(HieraPreTrainedModel):
    def __init__(self, config: HieraConfig) -> None:
        super().__init__(config)

        self.num_labels = config.num_labels
        self.hiera = HieraModel(config, add_pooling_layer=True, is_mae=False)

        # Classifier head
        self.classifier = (
            nn.Linear(self.hiera.num_features, config.num_labels) if config.num_labels > 0 else nn.Identity()
        )

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(HIERA_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=HieraForImageClassificationOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, HieraForImageClassificationOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        outputs = self.hiera(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[4:]
            return ((loss,) + output) if loss is not None else output

        return HieraForImageClassificationOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            reshaped_hidden_states=outputs.reshaped_hidden_states,
        )


@add_start_docstrings(
    """
    Hiera backbone, to be used with frameworks like DETR and MaskFormer.
    """,
    HIERA_START_DOCSTRING,
)
class HieraBackbone(HieraPreTrainedModel, BackboneMixin):
    def __init__(self, config: HieraConfig):
        super().__init__(config)
        super()._init_backbone(config)

        self.num_features = [config.embed_dim] + [
            int(config.embed_dim * config.embed_dim_multiplier**i) for i in range(len(config.depths))
        ]
        self.embeddings = HieraEmbeddings(config, is_mae=False)
        self.encoder = HieraEncoder(config)

        # Add layer norms to hidden states of out_features
        hidden_states_norms = {}
        for stage, num_channels in zip(self._out_features, self.channels):
            hidden_states_norms[stage] = nn.LayerNorm(num_channels)
        self.hidden_states_norms = nn.ModuleDict(hidden_states_norms)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    def forward(
        self,
        pixel_values: torch.Tensor,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> BackboneOutput:
        """
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, AutoBackbone
        >>> import torch
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> processor = AutoImageProcessor.from_pretrained("EduardoPacheco/hiera-tiny-224")
        >>> model = AutoBackbone.from_pretrained(
        ...     "EduardoPacheco/hiera-tiny-224", out_features=["stage1", "stage2", "stage3", "stage4"]
        ... )

        >>> inputs = processor(image, return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> feature_maps = outputs.feature_maps
        >>> list(feature_maps[-1].shape)
        [1, 768, 7, 7]
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        embedding_output, _, _ = self.embeddings(pixel_values)

        outputs = self.encoder(
            embedding_output,
            head_mask=None,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
        )

        hidden_states = outputs.reshaped_hidden_states

        feature_maps = ()
        for stage, hidden_state in zip(self.stage_names, hidden_states):
            if stage in self.out_features:
                batch_size, height, width, num_channels = hidden_state.shape
                hidden_state = hidden_state.view(batch_size, height * width, num_channels)
                hidden_state = self.hidden_states_norms[stage](hidden_state)
                hidden_state = hidden_state.view(batch_size, height, width, num_channels)
                hidden_state = hidden_state.permute(0, 3, 1, 2).contiguous()
                feature_maps += (hidden_state,)

        if not return_dict:
            output = (feature_maps,)
            if output_hidden_states:
                output += (outputs.hidden_states,)
            return output

        return BackboneOutput(
            feature_maps=feature_maps,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions,
        )
