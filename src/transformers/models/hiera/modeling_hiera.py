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
""" PyTorch Hiera model."""


import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    ImageClassifierOutput,
    MaskedImageModelingOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_hiera import HieraConfig


logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "HieraConfig"

# Base docstring
_CHECKPOINT_FOR_DOC = "EduardoPacheco/hiera-tiny-224"
_EXPECTED_OUTPUT_SHAPE = [1, 197, 768]

# Image classification docstring
_IMAGE_CLASS_CHECKPOINT = "google/hiera-base-patch16-224"
_IMAGE_CLASS_EXPECTED_OUTPUT = "Egyptian cat"


HIERA_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "EduardoPacheco/hiera-tiny-224",
    # See all Hiera models at https://huggingface.co/models?filter=hiera
]


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

    def __init__(self, config):
        super().__init__()

        # Support any number of spatial dimensions
        self.spatial_dims = len(config.patch_kernel)
        if self.spatial_dims not in (2, 3):
            raise ValueError(
                f"The number of dimensions of the input image should be 2 or 3, but got {self.spatial_dims}."
            )
        self.num_channels = config.num_channels
        self.image_size = config.input_size

        self.projection = conv_nd(self.spatial_dims)(
            self.num_channels,
            config.hidden_size,
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

        if len(mask.shape[2:]) != len(target_size):
            raise ValueError(
                f"The length of the spatial dimensions of the mask should match the one from input image, but got {len(mask.shape[2:])} and {len(target_size)}."
            )

        if mask.shape[2:] != target_size:
            mask = nn.functional.interpolate(mask.float(), size=target_size)

        return self.projection(pixel_values * mask.bool())

    def forward(self, pixel_values: torch.Tensor, bool_masked_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        _, num_channels, _, _ = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
                f" Expected {self.num_channels} but got {num_channels}."
            )

        embeddings = self.masked_conv(pixel_values, bool_masked_pos)
        embeddings = embeddings.reshape(embeddings.shape[0], embeddings.shape[1], -1).transpose(2, 1)

        return embeddings


class HieraEmbeddings(nn.Module):
    """
    Construct position and patch embeddings.
    """

    def __init__(self, config: HieraConfig, use_mask_token: bool = False) -> None:
        super().__init__()

        self.tokens_spatial_shape = [i // s for i, s in zip(config.input_size, config.patch_stride)]
        self.num_tokens = math.prod(self.tokens_spatial_shape)
        self.sep_pos_embed = config.sep_pos_embed
        self.mask_spatial_shape = [i // s for i, s in zip(self.tokens_spatial_shape, config.masked_unit_size)]

        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim)) if use_mask_token else None

        self.patch_embeddings = HieraPatchEmbeddings(config)

        if self.sep_pos_embed:
            self.position_embeddings_spatial = nn.Parameter(
                torch.zeros(
                    1,
                    self.tokens_spatial_shape[1] * self.tokens_spatial_shape[2],
                    config.hidden_size,
                )
            )
            self.position_embeddings_temporal = nn.Parameter(
                torch.zeros(1, self.tokens_spatial_shape[0], config.hidden_size)
            )
        else:
            self.position_embeddings = nn.Parameter(torch.zeros(1, self.num_tokens, config.hidden_size))

    def get_position_embedding(self) -> torch.Tensor:
        if self.sep_pos_embed:
            return self.position_embeddings_spatial.repeat(
                1, self.tokens_spatial_shape[0], 1
            ) + torch.repeat_interleave(
                self.position_embeddings_temporal,
                self.tokens_spatial_shape[1] * self.tokens_spatial_shape[2],
                dim=1,
            )
        else:
            return self.position_embeddings

    def forward(self, pixel_values: torch.Tensor, bool_masked_pos: Optional[torch.BoolTensor] = None) -> torch.Tensor:
        if len(self.mask_spatial_shape) == 2:
            batch_size, num_channels, height, width = pixel_values.shape
        else:
            batch_size, num_channels, depth, height, width = pixel_values.shape

        if bool_masked_pos is not None:
            bool_masked_pos = bool_masked_pos.view(batch_size, 1, *self.mask_spatial_shape)

        embeddings = self.patch_embeddings(pixel_values, bool_masked_pos=bool_masked_pos)

        embeddings = embeddings + self.get_position_embedding()

        return embeddings


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
        stage_num: int,
    ) -> None:
        super().__init__()
        # we need to know if the previous stage used masked attention
        # mask unit or global attention.
        # lag by 1 layer, so that global attention,
        # applied post pooling on lower resolution
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

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, stage_module in enumerate(self.stages):
            layer_head_mask = head_mask[i] if head_mask is not None else None
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

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

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class HieraUnroll(nn.Module):
    """
    Reorders the tokens such that patches are contiguous in memory.
    E.g., given [B, (H, W), C] and stride of (Sy, Sx), this will re-order the tokens as
                           [B, (Sy, Sx, H // Sy, W // Sx), C]

    This allows operations like Max2d to be computed as x.view(B, Sx*Sy, -1, C).max(dim=1).
    Not only is this faster, but it also makes it easy to support inputs of arbitrary
    dimensions in addition to patch-wise sparsity.

    Performing this operation multiple times in sequence puts entire windows as contiguous
    in memory. For instance, if you applied the stride (2, 2) 3 times, entire windows of
    size 8x8 would be contiguous in memory, allowing operations like mask unit attention
    computed easily and efficiently, while also allowing max to be applied sequentially.

    Note: This means that intermediate values of the model are not in HxW order, so they
    need to be re-rolled if you want to use the intermediate values as a HxW feature map.
    The last block of the network is fine though, since by then the strides are all consumed.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.size = [i // s for i, s in zip(config.input_size, config.patch_stride)]
        self.schedule = [config.query_stride] * len(config.depths[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: Flattened patch embeddings [B, N, C]
        Output: Patch embeddings [B, N, C] permuted such that [B, 4, N//4, C].max(1) etc. performs MaxPoolNd
        """
        B, _, C = x.shape

        cur_size = self.size
        x = x.view(*([B] + cur_size + [C]))

        for strides in self.schedule:
            # Move patches with the given strides to the batch dimension

            # Create a view of the tensor with the patch stride as separate dims
            # For example in 2d: [B, H // Sy, Sy, W // Sx, Sx, C]
            cur_size = [i // s for i, s in zip(cur_size, strides)]
            new_shape = [B] + sum([[i, s] for i, s in zip(cur_size, strides)], []) + [C]
            x = x.view(new_shape)

            # Move the patch stride into the batch dimension
            # For example in 2d: [B, Sy, Sx, H // Sy, W // Sx, C]
            L = len(new_shape)
            permute = [0] + list(range(2, L - 1, 2)) + list(range(1, L - 1, 2)) + [L - 1]
            x = x.permute(permute)

            # Now finally flatten the relevant dims into the batch dimension
            x = x.flatten(0, len(strides))
            B *= math.prod(strides)

        x = x.reshape(-1, math.prod(self.size), C)
        return x


def undo_windowing(x: torch.Tensor, shape: List[int], mu_shape: List[int]) -> torch.Tensor:
    """
    Restore spatial organization by undoing windowed organization of mask units.

    Args:
        x: organized by mask units windows, e.g. in 2d [B, #MUy*#MUx, MUy, MUx, C]
        shape: current spatial shape, if it were not organized into mask unit
            windows, e.g. in 2d [B, #MUy*MUy, #MUx*MUx, C].
        mu_shape: current mask unit shape, e.g. in 2d [MUy, MUx]
    Returns:
        x: e.g. in 2d, [B, #MUy*MUy, #MUx*MUx, C]
    """
    D = len(shape)
    B, C = x.shape[0], x.shape[-1]
    # [B, #MUy*#MUx, MUy, MUx, C] -> [B, #MUy, #MUx, MUy, MUx, C]
    num_MUs = [s // mu for s, mu in zip(shape, mu_shape)]
    x = x.view(B, *num_MUs, *mu_shape, C)

    # [B, #MUy, #MUx, MUy, MUx, C] -> [B, #MUy*MUy, #MUx*MUx, C]
    permute = (
        [0]
        + sum(
            [list(p) for p in zip(range(1, 1 + D), range(1 + D, 1 + 2 * D))],
            [],
        )
        + [len(x.shape) - 1]
    )
    x = x.permute(permute).reshape(B, *shape, C)

    return x


class HieraReroll(nn.Module):
    """
    Undos the "unroll" operation so that you can use intermediate features.
    """

    def __init__(
        self,
        input_size: Tuple[int, ...],
        patch_stride: Tuple[int, ...],
        unroll_schedule: List[Tuple[int, ...]],
        stage_ends: List[int],
        q_pool: int,
    ):
        super().__init__()
        self.size = [i // s for i, s in zip(input_size, patch_stride)]

        # The first stage has to reverse everything
        # The next stage has to reverse all but the first unroll, etc.
        self.schedule = {}
        size = self.size
        for i in range(stage_ends[-1] + 1):
            self.schedule[i] = unroll_schedule, size
            # schedule unchanged if no pooling at a stage end
            if i in stage_ends[:q_pool]:
                if len(unroll_schedule) > 0:
                    size = [n // s for n, s in zip(size, unroll_schedule[0])]
                unroll_schedule = unroll_schedule[1:]

    def forward(self, x: torch.Tensor, block_idx: int, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Roll the given tensor back up to spatial order assuming it's from the given block.

        If no mask is provided:
            - Returns [B, H, W, C] for 2d, [B, T, H, W, C] for 3d, etc.
        If a mask is provided:
            - Returns [B, #MUs, MUy, MUx, C] for 2d, etc.
        """
        schedule, size = self.schedule[block_idx]
        B, N, C = x.shape

        D = len(size)
        cur_mu_shape = [1] * D

        for strides in schedule:
            # Extract the current patch from N
            x = x.view(B, *strides, N // math.prod(strides), *cur_mu_shape, C)

            # Move that patch into the current MU
            # Example in 2d: [B, Sy, Sx, N//(Sy*Sx), MUy, MUx, C] -> [B, N//(Sy*Sx), Sy, MUy, Sx, MUx, C]
            L = len(x.shape)
            permute = (
                [0, 1 + D]
                + sum(
                    [list(p) for p in zip(range(1, 1 + D), range(1 + D + 1, L - 1))],
                    [],
                )
                + [L - 1]
            )
            x = x.permute(permute)

            # Reshape to [B, N//(Sy*Sx), *MU, C]
            for i in range(D):
                cur_mu_shape[i] *= strides[i]
            x = x.reshape(B, -1, *cur_mu_shape, C)
            N = x.shape[1]

        # Current shape (e.g., 2d: [B, #MUy*#MUx, MUy, MUx, C])
        x = x.view(B, N, *cur_mu_shape, C)

        # If masked, return [B, #MUs, MUy, MUx, C]
        if mask is not None:
            return x

        # If not masked, we can return [B, H, W, C]
        x = undo_windowing(x, size, cur_mu_shape)

        return x


class HieraPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = HieraConfig
    base_model_prefix = "hiera"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True
    _no_split_modules = ["HieraEmbeddings", "HieraLayer"]

    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]) -> None:
        """Initialize the weights"""
        std = self.config.initializer_range

        if isinstance(module, HieraEmbeddings):
            if self.config.sep_pos_embed:
                nn.init.trunc_normal_(module.position_embeddings_spatial, std=std)
                nn.init.trunc_normal_(module.position_embeddings_temporal, std=std)
            else:
                nn.init.trunc_normal_(module.position_embeddings, std=std)

        elif isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            nn.init.trunc_normal_(module.weight, std=std)
            if isinstance(module, nn.Linear) and module.bias is not None:
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
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See [`ViTImageProcessor.__call__`]
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


@add_start_docstrings(
    "The bare Hiera Model transformer outputting raw hidden-states without any specific head on top.",
    HIERA_START_DOCSTRING,
)
class HieraModel(HieraPreTrainedModel):
    def __init__(self, config: HieraConfig, add_pooling_layer: bool = True, use_mask_token: bool = False):
        super().__init__(config)
        self.config = config
        self.num_layers = len(config.depths)
        self.num_features = int(config.embed_dim * config.embed_dim_multiplier ** (self.num_layers - 1))

        self.embeddings = HieraEmbeddings(config, use_mask_token=use_mask_token)
        self.unroll = HieraUnroll(config)
        self.encoder = HieraEncoder(config)

        self.layernorm = nn.LayerNorm(self.num_features, eps=config.layer_norm_eps)
        self.pooler = nn.AdaptiveAvgPool1d(1) if add_pooling_layer else None

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
        output_type=BaseModelOutputWithPooling,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`, *optional*):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
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

        embedding_output = self.embeddings(pixel_values, bool_masked_pos=bool_masked_pos)

        hidden_states = self.unroll(embedding_output)

        encoder_outputs = self.encoder(
            hidden_states,
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
            pooled_output = self.layernorm(pooled_output)

        if not return_dict:
            head_outputs = (sequence_output, pooled_output) if pooled_output is not None else (sequence_output,)
            return head_outputs + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


@add_start_docstrings(
    """Hiera Model with a decoder on top for masked image modeling, as proposed in [SimMIM](https://arxiv.org/abs/2111.09886).

    <Tip>

    Note that we provide a script to pre-train this model on custom data in our [examples
    directory](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-pretraining).

    </Tip>
    """,
    HIERA_START_DOCSTRING,
)
# Copied from transformers.models.vit.modeling_vit.ViTForMaskedImageModeling with VIT->HIERA,ViT->Hiera,vit->hiera,google/vit-base-patch16-224-in21k->EduardoPacheco/hiera-tiny-224
class HieraForMaskedImageModeling(HieraPreTrainedModel):
    def __init__(self, config: HieraConfig) -> None:
        super().__init__(config)

        self.hiera = HieraModel(config, add_pooling_layer=False, use_mask_token=True)

        self.decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=config.hidden_size,
                out_channels=config.encoder_stride**2 * config.num_channels,
                kernel_size=1,
            ),
            nn.PixelShuffle(config.encoder_stride),
        )

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(HIERA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=MaskedImageModelingOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, MaskedImageModelingOutput]:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).

        Returns:

        Examples:
        ```python
        >>> from transformers import AutoImageProcessor, HieraForMaskedImageModeling
        >>> import torch
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("google/hiera-base-patch16-224-in21k")
        >>> model = HieraForMaskedImageModeling.from_pretrained("google/hiera-base-patch16-224-in21k")

        >>> num_patches = (model.config.image_size // model.config.patch_size) ** 2
        >>> pixel_values = image_processor(images=image, return_tensors="pt").pixel_values
        >>> # create random boolean mask of shape (batch_size, num_patches)
        >>> bool_masked_pos = torch.randint(low=0, high=2, size=(1, num_patches)).bool()

        >>> outputs = model(pixel_values, bool_masked_pos=bool_masked_pos)
        >>> loss, reconstructed_pixel_values = outputs.loss, outputs.reconstruction
        >>> list(reconstructed_pixel_values.shape)
        [1, 3, 224, 224]
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if bool_masked_pos is not None and (self.config.patch_size != self.config.encoder_stride):
            raise ValueError(
                "When `bool_masked_pos` is provided, `patch_size` must be equal to `encoder_stride` to ensure that "
                "the reconstructed image has the same dimensions as the input. "
                f"Got `patch_size` = {self.config.patch_size} and `encoder_stride` = {self.config.encoder_stride}."
            )

        outputs = self.hiera(
            pixel_values,
            bool_masked_pos=bool_masked_pos,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        # Reshape to (batch_size, num_channels, height, width)
        sequence_output = sequence_output[:, 1:]
        batch_size, sequence_length, num_channels = sequence_output.shape
        height = width = math.floor(sequence_length**0.5)
        sequence_output = sequence_output.permute(0, 2, 1).reshape(batch_size, num_channels, height, width)

        # Reconstruct pixel values
        reconstructed_pixel_values = self.decoder(sequence_output)

        masked_im_loss = None
        if bool_masked_pos is not None:
            size = self.config.image_size // self.config.patch_size
            bool_masked_pos = bool_masked_pos.reshape(-1, size, size)
            mask = (
                bool_masked_pos.repeat_interleave(self.config.patch_size, 1)
                .repeat_interleave(self.config.patch_size, 2)
                .unsqueeze(1)
                .contiguous()
            )
            reconstruction_loss = nn.functional.l1_loss(pixel_values, reconstructed_pixel_values, reduction="none")
            masked_im_loss = (reconstruction_loss * mask).sum() / (mask.sum() + 1e-5) / self.config.num_channels

        if not return_dict:
            output = (reconstructed_pixel_values,) + outputs[1:]
            return ((masked_im_loss,) + output) if masked_im_loss is not None else output

        return MaskedImageModelingOutput(
            loss=masked_im_loss,
            reconstruction=reconstructed_pixel_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    Hiera Model transformer with an image classification head on top (a linear layer on top of the final hidden state of
    the [CLS] token) e.g. for ImageNet.

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
        self.hiera = HieraModel(config)

        # Classifier head
        self.classifier = (
            nn.Linear(self.hiera.num_features, config.num_labels) if config.num_labels > 0 else nn.Identity()
        )

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(HIERA_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=ImageClassifierOutput,
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
    ) -> Union[tuple, ImageClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
