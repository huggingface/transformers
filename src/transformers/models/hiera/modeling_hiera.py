# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
#
# Hiera: A Hierarchical Vision Transformer without the Bells-and-Whistles
#
# Chaitanya Ryali, Yuan-Ting Hu, Daniel Bolya, Chen Wei, Haoqi Fan,
# Po-Yao Huang, Vaibhav Aggarwal, Arkabandhu Chowdhury, Omid Poursaeed,
# Judy Hoffman, Jitendra Malik, Yanghao Li, Christoph Feichtenhofer.
#
# Paper: https://arxiv.org/abs/2306.00989/
#
# References:
# slowfast: https://github.com/facebookresearch/SlowFast
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# --------------------------------------------------------

import collections.abc
import math
from dataclasses import dataclass
from functools import partial
from itertools import repeat
from typing import Callable, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...modeling_utils import PreTrainedModel
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)
from .configuration_hiera import HieraConfig


HIERA_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "namangarg110/hiera_base_224",
]


def conv_nd(n: int) -> Type[nn.Module]:
    """
    Returns a conv with nd (e.g., Conv2d for n=2). Work up to n=3.
    If you wanted a 4d HieraModel, you could probably just implement this for n=4. (no promises)
    """
    return [nn.Identity, nn.Conv1d, nn.Conv2d, nn.Conv3d][n]


def do_pool(x: torch.Tensor, stride: int) -> torch.Tensor:
    # Refer to `Unroll` to see how this performs a maxpool-Nd
    return x.view(x.shape[0], stride, -1, x.shape[-1]).max(dim=1).values


def get_resized_mask(target_size: torch.Size, mask: torch.Tensor) -> torch.Tensor:
    # target_size: [(T), (H), W]
    # (spatial) mask: [B, C, (t), (h), w]
    if mask is None:
        return mask

    assert len(mask.shape[2:]) == len(target_size)
    if mask.shape[2:] != target_size:
        return F.interpolate(mask.float(), size=target_size)
    return mask


def do_masked_conv(x: torch.Tensor, conv: nn.Module, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Zero-out the masked regions of the input before conv.
    Prevents leakage of masked regions when using overlapping kernels.
    """
    if conv is None:
        return x
    if mask is None:
        return conv(x)

    mask = get_resized_mask(target_size=x.shape[2:], mask=mask)
    return conv(x * mask.bool())


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


# Copied from transformers.models.swin.modeling_swin.drop_path
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


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob,3):0.3f}"


# Copied from timm.layers.helpers
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


# Copied from timm.layers.mlp
class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        norm_layer=None,
        bias=True,
        drop=0.0,
        use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Unroll(nn.Module):
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

    def __init__(
        self,
        input_size: Tuple[int, ...],
        patch_stride: Tuple[int, ...],
        unroll_schedule: List[Tuple[int, ...]],
    ):
        super().__init__()
        self.size = [i // s for i, s in zip(input_size, patch_stride)]
        self.schedule = unroll_schedule

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


class Reroll(nn.Module):
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
            Returns [B, H, W, C] for 2d, [B, T, H, W, C] for 3d, etc.
        If a mask is provided:
            Returns [B, #MUs, MUy, MUx, C] for 2d, etc.
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


@dataclass
class HieraModelOutput(ModelOutput):
    """
    Base class for HieraModel model's outputs, conforming to Hugging Face's ModelOutput.

    Args:
        last_hidden_state (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)):
            Last layer hidden-states.
        intermediates (List[torch.Tensor], optional):
            Intermediate representations or features from the model, if applicable.
    """

    last_hidden_state: torch.FloatTensor
    intermediates: Optional[List[torch.Tensor]] = None


class MaskUnitAttention(nn.Module):
    """
    Computes either Mask Unit or Global Attention. Also is able to perform q pooling.

    Note: this assumes the tokens have already been flattened and unrolled into mask units.
    See `Unroll` for more details.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        number_of_heads: int,
        q_stride: int = 1,
        window_size: int = 0,
        use_mask_unit_attention: bool = False,
    ):
        """
        Args:
            input_dim (`int`): The input feature dimensions.
            output_dim (`int`): The output feature dimensions.
            number_of_heads (`int`): The number of attention heads.
            q_stride: If greater than 1, pool q with this stride. The stride should be flattened (e.g., 2x2 = 4).
            window_size: The current (flattened) size of a mask unit *after* pooling (if any).
            use_mask_unit_attention: Use Mask Unit or Global Attention.
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.number_of_heads = number_of_heads
        self.q_stride = q_stride

        self.head_dim = output_dim // number_of_heads
        self.scale = (self.head_dim) ** -0.5

        self.qkv = nn.Linear(input_dim, 3 * output_dim)
        self.projection = nn.Linear(output_dim, output_dim)

        self.window_size = window_size
        self.use_mask_unit_attention = use_mask_unit_attention

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Input should be of shape [batch, tokens, channels]."""
        batch_size, num_channels, _ = embeddings.shape
        num_windows = (num_channels // (self.q_stride * self.window_size)) if self.use_mask_unit_attention else 1

        qkv = (
            self.qkv(embeddings)
            .reshape(batch_size, -1, num_windows, 3, self.number_of_heads, self.head_dim)
            .permute(3, 0, 4, 2, 1, 5)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.q_stride > 1:
            # Refer to Unroll to see how this performs a maxpool-Nd
            q = (
                q.view(batch_size, self.number_of_heads, num_windows, self.q_stride, -1, self.head_dim)
                .max(dim=3)
                .values
            )

        if hasattr(F, "scaled_dot_product_attention"):
            # Note: the original paper did *not* use SDPA, it's a free boost!
            embeddings = F.scaled_dot_product_attention(q, k, v)
        else:
            attention = (q * self.scale) @ k.transpose(-1, -2)
            attention = attention.softmax(dim=-1)
            embeddings = attention @ v

        embeddings = embeddings.transpose(1, 3).reshape(batch_size, -1, self.output_dim)
        embeddings = self.projection(embeddings)
        return embeddings


class HieraBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        number_of_heads: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        act_layer: nn.Module = nn.GELU,
        q_stride: int = 1,
        window_size: int = 0,
        use_mask_unit_attention: bool = False,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.norm1 = norm_layer(input_dim)
        self.attention = MaskUnitAttention(
            input_dim, output_dim, number_of_heads, q_stride, window_size, use_mask_unit_attention
        )

        self.norm2 = norm_layer(output_dim)
        self.mlp = Mlp(output_dim, int(output_dim * mlp_ratio), act_layer=act_layer)

        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        if input_dim != output_dim:
            self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        # Attention + Q Pooling
        normalized_embeddings = self.norm1(embeddings)
        if self.input_dim != self.output_dim:
            embeddings = do_pool(self.projection(normalized_embeddings), stride=self.attention.q_stride)
        embeddings = embeddings + self.drop_path(self.attention(normalized_embeddings))

        # MLP
        embeddings = embeddings + self.drop_path(self.mlp(self.norm2(embeddings)))
        return embeddings


class Head(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        dropout_rate: float = 0.0,
        act_func: Callable[[torch.Tensor], torch.Tensor] = lambda x: x.softmax(dim=-1),
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.projection = nn.Linear(input_dim, num_classes)
        # act_fun for eval and testing only
        self.act_func = act_func

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        x = self.projection(x)
        if not self.training:
            x = self.act_func(x)
        return x


@add_start_docstrings(
    """
    Patch embedding that supports any number of spatial dimensions (1d, 2d, 3d).
    """
)
class PatchEmbedding(nn.Module):
    def __init__(
        self,
        dim_in: int,
        output_dim: int,
        kernel: Tuple[int, ...],
        stride: Tuple[int, ...],
        padding: Tuple[int, ...],
    ):
        super().__init__()

        # Support any number of spatial dimensions
        self.spatial_dims = len(kernel)
        self.projection = conv_nd(self.spatial_dims)(
            dim_in,
            output_dim,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
        )

    def forward(self, pixel_values: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        embeddings = do_masked_conv(pixel_values, self.projection, mask)
        embeddings = embeddings.reshape(embeddings.shape[0], embeddings.shape[1], -1).transpose(2, 1)
        return embeddings


class HieraPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = HieraConfig
    base_model_prefix = "hiera"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True

    def _init_weights(self, module, init_bias=0.02):
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.constant_(module.bias, init_bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, init_bias)
            nn.init.constant_(module.weight, 1.0)


@add_start_docstrings(
    """
    Hiera: A Hierarchical Vision Transformer without the Bells-and-Whistles.

    This model is a PyTorch implementation of the Hiera architecture for image classification. It introduces a hierarchical design that processes images in a coarse-to-fine manner, efficiently handling various scales and complexities within the images.

    The model is built on the principles of Vision Transformers but introduces mask units to focus on specific regions of interest, significantly reducing computational requirements while maintaining competitive performance.

    Parameters:
        config ([`HieraConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.

    Example usage:
        >>> from transformers import HieraModel, HieraConfig
        >>> import torch
        >>> config = HieraConfig(embedding_dimension=96, number_of_heads=1, stages=(2, 3, 16, 3))
        >>> model = HieraModel(config)
        >>> inputs = torch.rand((1, 3, 224, 224))
        >>> outputs = model(inputs)
    """
)
class HieraModel(HieraPreTrainedModel):
    config_class = HieraConfig
    base_model_prefix = "hiera"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True

    def __init__(self, config: HieraConfig):
        self.input_size = config.input_size
        self.in_chans = config.in_chans
        self.embedding_dimension = config.embedding_dimension
        self.number_of_heads = config.number_of_heads
        self.num_classes = config.num_classes
        self.stages = config.stages
        self.q_pool = config.q_pool
        self.q_stride = config.q_stride
        self.mask_unit_size = config.mask_unit_size
        self.mask_unit_attn = config.mask_unit_attn
        self.dim_mul = config.dim_mul
        self.head_mul = config.head_mul
        self.patch_kernel = config.patch_kernel
        self.patch_stride = config.patch_stride
        self.patch_padding = config.patch_padding
        self.mlp_ratio = config.mlp_ratio
        self.drop_path_rate = config.drop_path_rate
        self.head_dropout = config.head_dropout
        self.head_init_scale = config.head_init_scale
        self.sep_position_embeddings = config.sep_position_embeddings

        super().__init__(config)
        self.config = config
        norm_layer = partial(nn.LayerNorm, eps=1e-6)  # Example, adjust as needed
        depth = sum(self.stages)
        self.tokens_spatial_shape = [i // s for i, s in zip(self.input_size, self.patch_stride)]
        num_tokens = math.prod(self.tokens_spatial_shape)
        flat_mu_size = math.prod(self.mask_unit_size)
        flat_q_stride = math.prod(self.q_stride)

        assert self.q_pool < len(self.stages)
        self.q_pool, self.q_stride = self.q_pool, self.q_stride
        self.mu_size, self.mask_unit_size = flat_mu_size, self.mask_unit_size
        self.mask_spatial_shape = [i // s for i, s in zip(self.tokens_spatial_shape, self.mask_unit_size)]
        self.stage_ends = [sum(self.stages[:i]) - 1 for i in range(1, len(self.stages) + 1)]

        self.patch_embedding = PatchEmbedding(
            self.in_chans, self.embedding_dimension, self.patch_kernel, self.patch_stride, self.patch_padding
        )

        if self.sep_position_embeddings:
            self.position_embeddings_spatial = nn.Parameter(
                torch.zeros(
                    1,
                    self.tokens_spatial_shape[1] * self.tokens_spatial_shape[2],
                    self.embedding_dimension,
                )
            )
            self.position_embeddings_temporal = nn.Parameter(
                torch.zeros(1, self.tokens_spatial_shape[0], self.embedding_dimension)
            )
        else:
            self.position_embeddings = nn.Parameter(torch.zeros(1, num_tokens, self.embedding_dimension))

        # Setup roll and reroll modules
        self.unroll = Unroll(self.input_size, self.patch_stride, [self.q_stride] * len(self.stage_ends[:-1]))
        self.reroll = Reroll(
            self.input_size,
            self.patch_stride,
            [self.q_stride] * len(self.stage_ends[:-1]),
            self.stage_ends,
            self.q_pool,
        )
        # q_pool locations
        q_pool_blocks = [x + 1 for x in self.stage_ends[: self.q_pool]]
        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, depth)]

        # Transformer blocks
        cur_stage = 0
        self.blocks = nn.ModuleList()

        for i in range(depth):
            output_dim = self.embedding_dimension
            # Mask unit or global attention.
            # Lag by 1 block, so that global attention,
            # applied post pooling on lower resolution
            use_mask_unit_attention = self.mask_unit_attn[cur_stage]

            if i - 1 in self.stage_ends:
                output_dim = int(self.embedding_dimension * self.dim_mul)
                number_of_heads = int(self.number_of_heads * self.head_mul)
                cur_stage += 1
                if i in q_pool_blocks:
                    flat_mu_size //= flat_q_stride
            else:
                number_of_heads = self.number_of_heads

            block = HieraBlock(
                input_dim=self.embedding_dimension,
                output_dim=output_dim,
                number_of_heads=number_of_heads,
                mlp_ratio=self.mlp_ratio,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                q_stride=(flat_q_stride if i in q_pool_blocks else 1),
                window_size=flat_mu_size,
                use_mask_unit_attention=use_mask_unit_attention,
            )

            self.embedding_dimension = output_dim
            self.blocks.append(block)

        self.norm = norm_layer(self.embedding_dimension)
        self.head = Head(self.embedding_dimension, self.num_classes, dropout_rate=self.head_dropout)

        # Initialize everything
        if self.sep_position_embeddings:
            nn.init.trunc_normal_(self.position_embeddings_spatial, std=0.02)
            nn.init.trunc_normal_(self.position_embeddings_temporal, std=0.02)
        else:
            nn.init.trunc_normal_(self.position_embeddings, std=0.02)
        self.apply(partial(self._init_weights))
        self.head.projection.weight.data.mul_(self.head_init_scale)
        self.head.projection.bias.data.mul_(self.head_init_scale)
        self.post_init()

    @torch.jit.ignore
    def no_weight_decay(self):
        if self.sep_position_embeddings:
            return ["position_embeddings_spatial", "position_embeddings_temporal"]
        else:
            return ["position_embeddings"]

    def get_random_mask(self, x: torch.Tensor, mask_ratio: float) -> torch.Tensor:
        """
        Generates a random mask, mask_ratio fraction are dropped.
        1 is *keep*, 0 is *remove*. Useful for MAE, FLIP, etc.
        """
        batch_size = x.shape[0]
        # Tokens selected for masking at mask unit level
        num_windows = math.prod(self.mask_spatial_shape)  # num_mask_units
        len_keep = int(num_windows * (1 - mask_ratio))
        noise = torch.rand(batch_size, num_windows, device=x.device)

        # Sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Generate the binary mask: 1 is *keep*, 0 is *remove*
        # Note this is opposite to original MAE
        mask = torch.zeros([batch_size, num_windows], device=x.device)
        mask[:, :len_keep] = 1
        # Unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return mask.bool()

    def get_position_embeddings(self) -> torch.Tensor:
        if self.sep_position_embeddings:
            return self.position_embeddings_spatial.repeat(
                1, self.tokens_spatial_shape[0], 1
            ) + torch.repeat_interleave(
                self.position_embeddings_temporal,
                self.tokens_spatial_shape[1] * self.tokens_spatial_shape[2],
                dim=1,
            )
        else:
            return self.position_embeddings

    @add_start_docstrings_to_model_forward(
        """
        The forward pass for the Hiera model.

        Args:
            pixel_values (`torch.Tensor`): Input tensor of shape `(batch_size, channels, height, width)`.

            mask (`torch.Tensor`, optional): A boolean tensor of shape `(batch_size, num_mask_units)` indicating which mask units to keep (True) or remove (False).
            mask should be a boolean tensor of shape [batch_size , #MUt*#MUy*#MUx] where #MU are the number of mask units in that input_dim.
            Note: 1 in mask is *keep*, 0 is *remove*; mask.sum(dim=-1) should be the same across the batch.
            return_dict (`bool`, optional): Whether to return a dictionary of outputs or a plain tuple.
            return_intermediates (`bool`, optional): Whether to return intermediate features from each stage of the model.
        """
    )
    def forward(
        self,
        pixel_values: torch.Tensor,
        mask: torch.Tensor = None,
        return_dict: Optional[bool] = True,
        return_intermediates: bool = True,
    ) -> Union[Tuple[torch.Tensor], HieraModelOutput]:
        # Slowfast training passes in a list
        if isinstance(pixel_values, list):
            pixel_values = pixel_values[0]
        intermediates = []

        pached_embeddings = self.patch_embedding(
            pixel_values,
            mask=mask.view(pixel_values.shape[0], 1, *self.mask_spatial_shape)  # batch_size , C, *mask_spatial_shape
            if mask is not None
            else None,
        )
        embeddings = pached_embeddings + self.get_position_embeddings()
        embeddings = self.unroll(embeddings)

        # Discard masked tokens
        if mask is not None:
            embeddings = embeddings[mask[..., None].tile(1, self.mu_size, embeddings.shape[2])].view(
                embeddings.shape[0], -1, embeddings.shape[-1]
            )

        for i, block in enumerate(self.blocks):
            embeddings = block(embeddings)

            if return_intermediates and i in self.stage_ends:
                intermediates.append(self.reroll(embeddings, i, mask=mask))

        if mask is None:
            embeddings = embeddings.mean(dim=1)
            embeddings = self.norm(embeddings)
            embeddings = self.head(embeddings)

        # embeddings may not always be in spatial order here.
        # e.g. if q_pool = 2, mask_unit_size = (8, 8), and
        # q_stride = (2, 2), not all unrolls were consumed,
        # intermediates[-1] is embeddings in spatial order
        if not return_dict:
            return tuple(v for v in [embeddings, intermediates] if v is not None)

        return HieraModelOutput(
            last_hidden_state=embeddings,
            intermediates=intermediates if return_intermediates else None,
        )
