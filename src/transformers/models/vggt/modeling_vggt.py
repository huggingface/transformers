# coding=utf-8
# Copyright 2021 Google AI, Ross Wightman, The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch VGGT model."""

import math
from typing import Optional, Tuple, Union, List

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch.nn.functional as F

from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    ImageClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from .configuration_vggt import VGGTConfig

# Import from layers directory
from .layers.attention import Attention
from .layers.block import Block
from .layers.patch_embed import PatchEmbed
from .layers.rope import RotaryPositionEmbedding2D, PositionGetter
from .layers.mlp import Mlp
from .layers.layer_scale import LayerScale
from .layers.drop_path import DropPath
from .layers.vision_transformer import vit_small, vit_base, vit_large, vit_giant2

# Import heads
from .heads.camera_head import CameraHead
from .heads.dpt_head import DPTHead
from .heads.track_head import TrackHead

logger = logging.get_logger(__name__)

# 添加必要的常量
VGGT_PRETRAINED_MODEL_ARCHIVE_LIST = []

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


def slice_expand_and_flatten(token_tensor, B, S):
    """
    Process specialized tokens for multi-frame sequences.
    References: aggregator.py slice_expand_and_flatten function
    """
    # Use first token for first frame, second token for remaining frames
    query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
    key = token_tensor[:, 1:2, ...].expand(B, S - 1, *token_tensor.shape[2:])
    
    # Concatenate and flatten
    tokens = torch.cat([query, key], dim=1)  # (B, S, X, C)
    return tokens.view(B * S, *tokens.shape[2:])  # (B*S, X, C)


# 添加基类
class VGGTPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = VGGTConfig
    base_model_prefix = "vggt"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Aggregator", "Block"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=self.config.init_values)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Parameter):
            module.data.normal_(mean=0.0, std=self.config.init_values)


class Aggregator(nn.Module):
    """
    The Aggregator applies alternating-attention over input frames,
    as described in VGGT: Visual Geometry Grounded Transformer.
    完全按照原始aggregator.py的结构实现
    """

    def __init__(self, config: VGGTConfig):
        super().__init__()
        
        # 使用config中的参数，保持与原始Aggregator一致的命名和结构
        img_size = config.img_size
        patch_size = config.patch_size
        embed_dim = config.embed_dim
        depth = config.depth
        num_heads = config.num_heads
        mlp_ratio = config.mlp_ratio
        num_register_tokens = config.num_register_tokens
        qkv_bias = config.qkv_bias
        proj_bias = config.proj_bias
        ffn_bias = config.ffn_bias
        patch_embed = config.patch_embed
        aa_order = config.aa_order
        aa_block_size = config.aa_block_size
        qk_norm = config.qk_norm
        rope_freq = config.rope_freq
        init_values = config.init_values

        self.__build_patch_embed__(patch_embed, img_size, patch_size, num_register_tokens, embed_dim=embed_dim)

        # Initialize rotary position embedding if frequency > 0
        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None

        self.frame_blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.global_blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,  # 修复：global_blocks也应该有rope
                )
                for _ in range(depth)
            ]
        )

        self.depth = depth
        self.aa_order = aa_order
        self.patch_size = patch_size
        self.aa_block_size = aa_block_size

        # Validate that depth is divisible by aa_block_size
        if self.depth % self.aa_block_size != 0:
            raise ValueError(f"depth ({depth}) must be divisible by aa_block_size ({aa_block_size})")

        self.aa_block_num = self.depth // self.aa_block_size

        # Note: We have two camera tokens, one for the first frame and one for the rest
        # The same applies for register tokens
        self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, embed_dim))

        # The patch tokens start after the camera and register tokens
        self.patch_start_idx = 1 + num_register_tokens

        # Initialize parameters with small values
        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)

        # Register normalization constants as buffers
        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 1, 3, 1, 1), persistent=False)

        self.use_reentrant = False  # hardcoded to False

    def __build_patch_embed__(
        self,
        patch_embed,
        img_size,
        patch_size,
        num_register_tokens,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        block_chunks=0,
        init_values=1.0,
        embed_dim=1024,
    ):
        """
        Build the patch embed layer. If 'conv', we use a
        simple PatchEmbed conv layer. Otherwise, we use a vision transformer.
        References: aggregator.py __build_patch_embed__
        """

        if "conv" in patch_embed:
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
        else:
            # DINOViT models mapping - references: aggregator.py
            vit_models = {
                "dinov2_vitl14_reg": vit_large,
                "dinov2_vitb14_reg": vit_base,
                "dinov2_vits14_reg": vit_small,
                "dinov2_vitg2_reg": vit_giant2,
            }

            self.patch_embed = vit_models[patch_embed](
                img_size=img_size,
                patch_size=patch_size,
                num_register_tokens=num_register_tokens,
                interpolate_antialias=interpolate_antialias,
                interpolate_offset=interpolate_offset,
                block_chunks=block_chunks,
                init_values=init_values,
            )

            # Disable gradient updates for mask token
            if hasattr(self.patch_embed, "mask_token"):
                self.patch_embed.mask_token.requires_grad_(False)

    def forward(self, images: torch.Tensor) -> Tuple[List[torch.Tensor], int]:
        """
        Forward pass of VGGT embeddings.
        
        Args:
            images (torch.Tensor): Input images with shape [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width

        Returns:
            (list[torch.Tensor], int):
                The list of outputs from the attention blocks,
                and the patch_start_idx indicating where patch tokens begin.
        """
        B, S, C_in, H, W = images.shape

        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        # Normalize images and reshape for patch embed
        images = (images - self._resnet_mean) / self._resnet_std

        # Reshape to [B*S, C, H, W] for patch embedding
        images = images.view(B * S, C_in, H, W)
        patch_tokens = self.patch_embed(images)

        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        _, P, C = patch_tokens.shape

        # Expand camera and register tokens to match batch size and sequence length
        camera_token = slice_expand_and_flatten(self.camera_token, B, S)
        register_token = slice_expand_and_flatten(self.register_token, B, S)

        # Concatenate special tokens with patch tokens
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)

        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=images.device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2).to(images.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        # update P because we added special tokens
        _, P, C = tokens.shape

        frame_idx = 0
        global_idx = 0
        output_list = []

        for _ in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos
                    )
                elif attn_type == "global":
                    tokens, global_idx, global_intermediates = self._process_global_attention(
                        tokens, B, S, P, C, global_idx, pos=pos
                    )
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            for i in range(len(frame_intermediates)):
                # concat frame and global intermediates, [B x S x P x 2C]
                concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                output_list.append(concat_inter)

        del concat_inter
        del frame_intermediates
        del global_intermediates
        return output_list, self.patch_start_idx

    def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None):
        """
        Process frame attention blocks. We keep tokens in shape (B*S, P, C).
        """
        # If needed, reshape tokens or positions:
        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B, S, P, C).view(B * S, P, C)

        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.view(B, S, P, 2).view(B * S, P, 2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            if self.training:
                tokens = checkpoint(self.frame_blocks[frame_idx], tokens, pos, use_reentrant=self.use_reentrant)
            else:
                tokens = self.frame_blocks[frame_idx](tokens, pos=pos)
            frame_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, frame_idx, intermediates

    def _process_global_attention(self, tokens, B, S, P, C, global_idx, pos=None):
        """
        Process global attention blocks. We keep tokens in shape (B, S*P, C).
        """
        if tokens.shape != (B, S * P, C):
            tokens = tokens.view(B, S, P, C).view(B, S * P, C)

        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.view(B, S, P, 2).view(B, S * P, 2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            if self.training:
                tokens = checkpoint(self.global_blocks[global_idx], tokens, pos, use_reentrant=self.use_reentrant)
            else:
                tokens = self.global_blocks[global_idx](tokens, pos=pos)
            global_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, global_idx, intermediates


class VGGTModel(VGGTPreTrainedModel):
    """
    VGGT backbone model for multi-frame vision tasks.
    与原始VGGT结构完全一致
    """
    
    config_class = VGGTConfig
    base_model_prefix = "vggt"
    main_input_name = "images"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Aggregator"]

    def __init__(self, config: VGGTConfig) -> None:
        super().__init__(config)
        self.config = config
        
        # 核心组件：只有aggregator，与原始结构一致
        self.aggregator = Aggregator(config)
        
        # Initialize weights
        self.post_init()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, Aggregator):
            # Aggregator内部处理gradient checkpointing
            pass

    def forward(
        self,
        images: torch.Tensor,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        """
        Forward pass of VGGT model.
        
        Args:
            images: Input images [B, S, 3, H, W] or [S, 3, H, W]
            output_hidden_states: Whether to return hidden states
            return_dict: Whether to return ModelOutput
            
        Returns:
            BaseModelOutput with aggregated_tokens_list and patch_start_idx
        """
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Handle input dimensions
        if len(images.shape) == 4:  # [S, 3, H, W]
            images = images.unsqueeze(0)  # [1, S, 3, H, W]
        
        # Aggregator forward
        aggregated_tokens_list, patch_start_idx = self.aggregator(images)
        
        if not return_dict:
            return (aggregated_tokens_list, patch_start_idx)
        
        # 创建BaseModelOutput，但添加自定义属性
        output = BaseModelOutput(
            last_hidden_state=aggregated_tokens_list[-1] if aggregated_tokens_list else None,
            hidden_states=tuple(aggregated_tokens_list) if output_hidden_states else None,
        )
        
        # 添加VGGT特有的属性
        output.aggregated_tokens_list = aggregated_tokens_list
        output.patch_start_idx = patch_start_idx
        
        return output


class VGGTForMultiTask(VGGTPreTrainedModel):
    """
    VGGT model with multiple task heads (camera, depth, point, tracking).
    与原始VGGT类完全一致的结构
    """
    
    config_class = VGGTConfig
    base_model_prefix = "vggt"
    main_input_name = "images"

    def __init__(
        self, 
        config: VGGTConfig,
        enable_camera: bool = True,
        enable_point: bool = True, 
        enable_depth: bool = True,
        enable_track: bool = True
    ) -> None:
        super().__init__(config)
        self.config = config
        
        # Backbone - 使用aggregator而不是分离的embedding和encoder
        self.aggregator = Aggregator(config)
        
        # Task heads - 与原始vggt.py完全一致
        self.camera_head = CameraHead(dim_in=2 * config.embed_dim) if enable_camera else None
        self.point_head = DPTHead(
            dim_in=2 * config.embed_dim, 
            output_dim=4, 
            activation="inv_log", 
            conf_activation="expp1"
        ) if enable_point else None
        self.depth_head = DPTHead(
            dim_in=2 * config.embed_dim, 
            output_dim=2, 
            activation="exp", 
            conf_activation="expp1"
        ) if enable_depth else None
        self.track_head = TrackHead(
            dim_in=2 * config.embed_dim, 
            patch_size=config.patch_size
        ) if enable_track else None
        
        # Initialize weights
        self.post_init()

    def forward(
        self,
        images: torch.Tensor,
        query_points: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> dict:
        """
        Forward pass with multiple task outputs.
        与原始vggt.py的forward方法完全一致
        """
        # Get backbone outputs - 直接使用aggregator
        aggregated_tokens_list, patch_start_idx = self.aggregator(images)
        
        predictions = {}
        
        # Handle input dimensions for heads
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        
        # Task predictions - 与原始实现完全一致
        with torch.cuda.amp.autocast(enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]
                predictions["pose_enc_list"] = pose_enc_list
                
            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

        if self.track_head is not None and query_points is not None:
            if len(query_points.shape) == 2:
                query_points = query_points.unsqueeze(0)
                
            track_list, vis, conf = self.track_head(
                aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, query_points=query_points
            )
            predictions["track"] = track_list[-1]
            predictions["vis"] = vis
            predictions["conf"] = conf

        if not self.training:
            predictions["images"] = images
            
        return predictions


# 添加到文件末尾
__all__ = [
    "VGGT_PRETRAINED_MODEL_ARCHIVE_LIST",
    "Aggregator",
    "VGGTModel",
    "VGGTPreTrainedModel",
    "VGGTForMultiTask",
]