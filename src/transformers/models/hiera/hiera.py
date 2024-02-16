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

import math
from functools import partial
from typing import List, Tuple, Callable, Optional
from .configuration_hiera import HieraConfig
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath, Mlp

from .hiera_utils import pretrained_model, conv_nd, do_pool, do_masked_conv, Unroll, Reroll



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
        - input_dim, output_dim: The input and output feature dimensions.
        - number_of_heads: The number of attention number_of_heads.
        - q_stride: If greater than 1, pool q with this stride. The stride should be flattened (e.g., 2x2 = 4).
        - window_size: The current (flattened) size of a mask unit *after* pooling (if any).
        - use_mask_unit_attention: Use Mask Unit or Global Attention.
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Input should be of shape [batch, tokens, channels]. """
        batch_size , num_channels , _ = x.shape
        num_windows = (
            (num_channels  // (self.q_stride * self.window_size)) if self.use_mask_unit_attention else 1
        )

        qkv = (
            self.qkv(x)
            .reshape(batch_size , -1, num_windows, 3, self.number_of_heads, self.head_dim)
            .permute(3, 0, 4, 2, 1, 5)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.q_stride > 1:
            # Refer to Unroll to see how this performs a maxpool-Nd
            q = (
                q.view(batch_size , self.number_of_heads, num_windows, self.q_stride, -1, self.head_dim)
                .max(dim=3)
                .values
            )

        if hasattr(F, "scaled_dot_product_attention"):
            # Note: the original paper did *not* use SDPA, it's a free boost!
            x = F.scaled_dot_product_attention(q, k, v)
        else:
            attention = (q * self.scale) @ k.transpose(-1, -2)
            attention = attention.softmax(dim=-1)
            x = (attention @ v)

        x = x.transpose(1, 3).reshape(batch_size , -1, self.output_dim)
        x = self.projection(x)
        return x


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention + Q Pooling
        normalized_input = self.norm1(x)
        if self.input_dim != self.output_dim:
            x = do_pool(self.projection(normalized_input), stride=self.attention.q_stride)
        x = x + self.drop_path(self.attention(normalized_input))

        # MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


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


class PatchEmbedding(nn.Module):
    """Patch embedding that supports any number of spatial dimensions (1d, 2d, 3d)."""

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

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = do_masked_conv(x, self.projection, mask)
        x = x.reshape(x.shape[0], x.shape[1], -1).transpose(2, 1)
        return x


class Hiera(nn.Module):
    def __init__(self, config: HieraConfig):
        super().__init__()
        self.config = config
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)  # Example, adjust as needed
        self.config = config
        depth = sum(self.config.stages)
        self.tokens_spatial_shape = [i // s for i, s in zip(self.config.input_size, self.config.patch_stride)]
        num_tokens = math.prod(self.tokens_spatial_shape)
        flat_mu_size = math.prod(self.config.mask_unit_size)
        flat_q_stride = math.prod(self.config.q_stride)

        assert self.config.q_pool < len(self.config.stages)
        self.q_pool, self.q_stride = self.config.q_pool, self.config.q_stride
        self.mu_size, self.mask_unit_size = flat_mu_size, self.config.mask_unit_size
        self.mask_spatial_shape = [
            i // s for i, s in zip(self.tokens_spatial_shape, self.mask_unit_size)
        ]
        self.stage_ends = [sum(self.config.stages[:i]) - 1 for i in range(1, len(self.config.stages) + 1)]

        self.patch_embedding = PatchEmbedding(
            self.config.in_chans, self.config.embedding_dimension, self.config.patch_kernel, self.config.patch_stride, self.config.patch_padding
        )

        if self.config.sep_position_embeddings:
            self.position_embeddings_spatial = nn.Parameter(
                torch.zeros(
                    1,
                    self.tokens_spatial_shape[1] * self.tokens_spatial_shape[2],
                    self.config.embedding_dimension,
                )
            )
            self.position_embeddings_temporal = nn.Parameter(
                torch.zeros(1, self.tokens_spatial_shape[0], self.config.embedding_dimension)
            )
        else:
            self.position_embeddings = nn.Parameter(torch.zeros(1, num_tokens, self.config.embedding_dimension))

        # Setup roll and reroll modules
        self.unroll = Unroll(
            self.config.input_size, self.config.patch_stride, [self.config.q_stride] * len(self.stage_ends[:-1])
        )
        self.reroll = Reroll(
            self.config.input_size,
            self.config.patch_stride,
            [self.config.q_stride] * len(self.stage_ends[:-1]),
            self.stage_ends,
            self.config.q_pool,
        )
        # q_pool locations
        q_pool_blocks = [x + 1 for x in self.stage_ends[:self.config.q_pool]]
        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, self.config.drop_path_rate, depth)]

        # Transformer blocks
        cur_stage = 0
        self.blocks = nn.ModuleList()

        for i in range(depth):
            output_dim = self.config.embedding_dimension
            # Mask unit or global attention.
            # Lag by 1 block, so that global attention,
            # applied post pooling on lower resolution
            use_mask_unit_attention = self.config.mask_unit_attn[cur_stage]

            if i - 1 in self.stage_ends:
                output_dim = int(self.config.embedding_dimension * self.config.dim_mul)
                number_of_heads = int(self.config.number_of_heads * self.config.head_mul)
                cur_stage += 1
                if i in q_pool_blocks:
                    flat_mu_size //= flat_q_stride
            else:
                number_of_heads = self.config.number_of_heads

            block = HieraBlock(
                input_dim=self.config.embedding_dimension,
                output_dim=output_dim,
                number_of_heads=number_of_heads,
                mlp_ratio=self.config.mlp_ratio,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                q_stride=(flat_q_stride if i in q_pool_blocks else 1),
                window_size=flat_mu_size,
                use_mask_unit_attention=use_mask_unit_attention,
            )

            self.config.embedding_dimension = output_dim
            self.blocks.append(block)

        self.norm = norm_layer(self.config.embedding_dimension)
        self.head = Head(self.config.embedding_dimension, self.config.num_classes, dropout_rate=self.config.head_dropout)

        # Initialize everything
        if self.config.sep_position_embeddings:
            nn.init.trunc_normal_(self.position_embeddings_spatial, std=0.02)
            nn.init.trunc_normal_(self.position_embeddings_temporal, std=0.02)
        else:
            nn.init.trunc_normal_(self.position_embeddings, std=0.02)
        self.apply(partial(self._init_weights))
        self.head.projection.weight.data.mul_(self.config.head_init_scale)
        self.head.projection.bias.data.mul_(self.config.head_init_scale)

    def _init_weights(self, m, init_bias=0.02):
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, init_bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, init_bias)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        if self.config.sep_position_embeddings:
            return ["position_embeddings_spatial", "position_embeddings_temporal"]
        else:
            return ["position_embeddings"]

    def get_random_mask(self, x: torch.Tensor, mask_ratio: float) -> torch.Tensor:
        """
        Generates a random mask, mask_ratio fraction are dropped.
        1 is *keep*, 0 is *remove*. Useful for MAE, FLIP, etc.
        """
        batch_size  = x.shape[0]
        # Tokens selected for masking at mask unit level
        num_windows = math.prod(self.mask_spatial_shape)  # num_mask_units
        len_keep = int(num_windows * (1 - mask_ratio))
        noise = torch.rand(batch_size , num_windows, device=x.device)

        # Sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Generate the binary mask: 1 is *keep*, 0 is *remove*
        # Note this is opposite to original MAE
        mask = torch.zeros([batch_size , num_windows], device=x.device)
        mask[:, :len_keep] = 1
        # Unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return mask.bool()

    def get_position_embeddings(self) -> torch.Tensor:
        if self.config.sep_position_embeddings:
            return self.position_embeddings_spatial.repeat(
                1, self.tokens_spatial_shape[0], 1
            ) + torch.repeat_interleave(
                self.position_embeddings_temporal,
                self.tokens_spatial_shape[1] * self.tokens_spatial_shape[2],
                dim=1,
            )
        else:
            return self.position_embeddings

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        return_intermediates: bool = False,
    ) -> torch.Tensor:
        """
        mask should be a boolean tensor of shape [batch_size , #MUt*#MUy*#MUx] where #MU are the number of mask units in that input_dim.
        Note: 1 in mask is *keep*, 0 is *remove*; mask.sum(dim=-1) should be the same across the batch.
        """
        # Slowfast training passes in a list
        if isinstance(x, list):
            x = x[0]
        intermediates = []

        x = self.patch_embedding(
            x,
            mask=mask.view(
                x.shape[0], 1, *self.mask_spatial_shape
            )  # batch_size , C, *mask_spatial_shape
            if mask is not None
            else None,
        )
        x = x + self.get_position_embeddings()
        x = self.unroll(x)

        # Discard masked tokens
        if mask is not None:
            x = x[mask[..., None].tile(1, self.mu_size, x.shape[2])].view(
                x.shape[0], -1, x.shape[-1]
            )

        for i, blk in enumerate(self.blocks):
            x = blk(x)

            if return_intermediates and i in self.stage_ends:
                intermediates.append(self.reroll(x, i, mask=mask))

        if mask is None:
            x = x.mean(dim=1)
            x = self.norm(x)
            x = self.head(x)

        # x may not always be in spatial order here.
        # e.g. if q_pool = 2, mask_unit_size = (8, 8), and
        # q_stride = (2, 2), not all unrolls were consumed,
        # intermediates[-1] is x in spatial order
        if return_intermediates:
            return x, intermediates

        return x


# Image models

@pretrained_model({
    "mae_in1k_ft_in1k": "https://dl.fbaipublicfiles.com/hiera/hiera_tiny_224.pth",
    "mae_in1k": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_tiny_224.pth",
}, default="mae_in1k_ft_in1k")
def hiera_tiny_224(**kwargs):
    config = HieraConfig(embedding_dimension=96, number_of_heads=1, stages=(1, 2, 7, 2), **kwargs)
    return Hiera(config)


@pretrained_model({
    "mae_in1k_ft_in1k": "https://dl.fbaipublicfiles.com/hiera/hiera_small_224.pth",
    "mae_in1k": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_small_224.pth",
}, default="mae_in1k_ft_in1k")
def hiera_small_224(**kwdargs):
    return Hiera(embedding_dimension=96, number_of_heads=1, stages=(1, 2, 11, 2), **kwdargs)


@pretrained_model({
    "mae_in1k_ft_in1k": "https://dl.fbaipublicfiles.com/hiera/hiera_base_224.pth",
    "mae_in1k": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_base_224.pth",
}, default="mae_in1k_ft_in1k")
def hiera_base_224(**kwargs):
    config = HieraConfig(embedding_dimention=96, number_of_heads=1, stages=(2, 3, 16, 3), **kwargs)
    return Hiera(config)


@pretrained_model({
    "mae_in1k_ft_in1k": "https://dl.fbaipublicfiles.com/hiera/hiera_base_plus_224.pth",
    "mae_in1k": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_base_plus_224.pth",
}, default="mae_in1k_ft_in1k")
def hiera_base_plus_224(**kwdargs):
    return Hiera(embedding_dimension=112, number_of_heads=2, stages=(2, 3, 16, 3), **kwdargs)


@pretrained_model({
    "mae_in1k_ft_in1k": "https://dl.fbaipublicfiles.com/hiera/hiera_large_224.pth",
    "mae_in1k": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_large_224.pth",
}, default="mae_in1k_ft_in1k")
def hiera_large_224(**kwdargs):
    return Hiera(embedding_dimension=144, number_of_heads=2, stages=(2, 6, 36, 4), **kwdargs)


@pretrained_model({
    "mae_in1k_ft_in1k": "https://dl.fbaipublicfiles.com/hiera/hiera_huge_224.pth",
    "mae_in1k": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_huge_224.pth",
}, default="mae_in1k_ft_in1k")
def hiera_huge_224(**kwdargs):
    return Hiera(embedding_dimension=256, number_of_heads=4, stages=(2, 6, 36, 4), **kwdargs)


# Video models

@pretrained_model({
    "mae_k400_ft_k400": "https://dl.fbaipublicfiles.com/hiera/hiera_base_16x224.pth",
    "mae_k400": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_base_16x224.pth",
}, default="mae_k400_ft_k400")
def hiera_base_16x224(num_classes: int = 400, **kwdargs):
    return Hiera(
        num_classes=num_classes,  # K400 has 400 classes
        input_size=(16, 224, 224),
        q_stride=(1, 2, 2),
        mask_unit_size=(1, 8, 8),
        patch_kernel=(3, 7, 7),
        patch_stride=(2, 4, 4),
        patch_padding=(1, 3, 3),
        sep_position_embeddings=True,
        **kwdargs
    )


@pretrained_model({
    "mae_k400_ft_k400": "https://dl.fbaipublicfiles.com/hiera/hiera_base_plus_16x224.pth",
    "mae_k400": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_base_plus_16x224.pth",
}, default="mae_k400_ft_k400")
def hiera_base_plus_16x224(**kwdargs):
    return hiera_base_16x224(
        embedding_dimension=112, number_of_heads=2, stages=(2, 3, 16, 3), **kwdargs
    )


@pretrained_model({
    "mae_k400_ft_k400": "https://dl.fbaipublicfiles.com/hiera/hiera_large_16x224.pth",
    "mae_k400": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_large_16x224.pth",
}, default="mae_k400_ft_k400")
def hiera_large_16x224(**kwdargs):
    return hiera_base_16x224(
        embedding_dimension=144, number_of_heads=2, stages=(2, 6, 36, 4), **kwdargs
    )


@pretrained_model({
    "mae_k400_ft_k400": "https://dl.fbaipublicfiles.com/hiera/hiera_huge_16x224.pth",
    "mae_k400": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_huge_16x224.pth",
}, default="mae_k400_ft_k400")
def hiera_huge_16x224(**kwdargs):
    return hiera_base_16x224(
        embedding_dimension=256, number_of_heads=4, stages=(2, 6, 36, 4), **kwdargs
    )
