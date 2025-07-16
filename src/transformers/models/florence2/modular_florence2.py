# coding=utf-8
# Copyright 2025 Microsoft and the HuggingFace Team. All rights reserved.
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
import math
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.activations import ACT2FN

from ...cache_utils import Cache
from ...configuration_utils import PretrainedConfig
from ...generation import GenerationMixin
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import Seq2SeqLMOutput, Seq2SeqModelOutput
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import (
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
    is_torchdynamo_compiling,
    logging,
)
from ..auto import CONFIG_MAPPING, AutoConfig, AutoModel
from ..bart.modeling_bart import eager_attention_forward
from ..beit.modeling_beit import BeitDropPath
from ..detr.modeling_detr import DetrLearnedPositionEmbedding
from ..llava.modeling_llava import LlavaPreTrainedModel


logger = logging.get_logger(__name__)


class Florence2VisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Florence2VisionModel`]. It is used to instantiate a Florence2VisionModel
    according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the Florence2VisionModel architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        in_channels (`int`, *optional*, defaults to 3):
            Number of input image channels.
        depths (`Tuple[int]`, *optional*, defaults to `(1, 1, 9, 1)`):
            The depth of the model.
        patch_size (`Tuple[int]`, *optional*, defaults to `(7, 3, 3, 3)`):
            The patch size of the image.
        patch_stride (`Tuple[int]`, *optional*, defaults to `(4, 2, 2, 2)`):
            The patch stride of the image.
        patch_padding (`Tuple[int]`, *optional*, defaults to `(3, 1, 1, 1)`):
            The patch padding of the image.
        patch_prenorm (`Tuple[bool]`, *optional*, defaults to `(False, True, True, True)`):
            Whether to apply layer normalization before the patch embedding layer.
        embed_dim (`Tuple[int]`, *optional*, defaults to `(128, 256, 512, 1024)`):
            The dimension of the embedding layer.
        num_heads (`Tuple[int]`, *optional*, defaults to `(4, 8, 16, 32)`):
            The number of attention heads.
        num_groups (`Tuple[int]`, *optional*, defaults to `(4, 8, 16, 32)`):
            The number of groups.
        window_size (`int`, *optional*, defaults to 12):
            The window size of the model.
        drop_path_rate (`float`, *optional*, defaults to 0.1):
            The dropout rate of the drop path layer.
        mlp_ratio (`int`, *optional*, defaults to 4.0):
            Ratio of mlp hidden dim to embedding dim.
        qkv_bias (`bool`, *optional*, defaults to `True`):
            If True, add a learnable bias to query, key, value.
        activation_function (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        projection_dim (`int`, *optional*, defaults to 1024):
            The dimension of the projection layer.
        max_temporal_embeddings (`int`, *optional*, defaults to 100):
            The configuration of the visual temporal embedding.
        max_position_embeddings (`int`, *optional*, defaults to 50):
            The configuration of the image position embedding.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    Example:

    ```python
    >>> from transformers import Florence2VisionConfig, Florence2VisionModel

    >>> # Initializing a Florence2 Vision style configuration
    >>> configuration = Florence2VisionConfig()

    >>> # Initializing a model (with random weights)
    >>> model = Florence2VisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "florence_vision"

    def __init__(
        self,
        in_channels=3,
        depths=(1, 1, 9, 1),
        patch_size=(7, 3, 3, 3),
        patch_stride=(4, 2, 2, 2),
        patch_padding=(3, 1, 1, 1),
        patch_prenorm=(False, True, True, True),
        embed_dim=(128, 256, 512, 1024),
        num_heads=(4, 8, 16, 32),
        num_groups=(4, 8, 16, 32),
        window_size=12,
        drop_path_rate=0.1,
        mlp_ratio=4.0,
        qkv_bias=True,
        activation_function="gelu",
        projection_dim=1024,
        max_temporal_embeddings=100,
        max_position_embeddings=50,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.depths = list(depths)
        self.patch_size = list(patch_size)
        self.patch_stride = list(patch_stride)
        self.patch_padding = list(patch_padding)
        self.patch_prenorm = list(patch_prenorm)
        self.embed_dim = list(embed_dim)
        self.num_heads = list(num_heads)
        self.num_groups = list(num_groups)
        self.window_size = window_size
        self.drop_path_rate = drop_path_rate
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.projection_dim = projection_dim
        self.max_temporal_embeddings = max_temporal_embeddings
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.activation_function = activation_function


class Florence2Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Florence2ForConditionalGeneration`]. It is used to instantiate an
    Florence-2 model according to the specified arguments, defining the model architecture.

    Instantiating a configuration with the defaults will yield a similar configuration to that of the Florence-2
    [microsoft/Florence-2-base](https://huggingface.co/microsoft/Florence-2-base) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`AutoConfig`].
        vision_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`Florence2VisionConfig`].

    Example:

    ```python
    >>> from transformers import Florence2ForConditionalGeneration, Florence2Config, CLIPVisionConfig, BartConfig

    >>> # Initializing a clip-like vision config
    >>> vision_config = CLIPVisionConfig()

    >>> # Initializing a Bart config
    >>> text_config = BartConfig()

    >>> # Initializing a Florence-2 configuration
    >>> configuration = Florence2Config(vision_config, text_config)

    >>> # Initializing a model from the florence-2 configuration
    >>> model = Florence2ForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "florence2"
    sub_configs = {
        "text_config": AutoConfig,
        "vision_config": Florence2VisionConfig,
    }

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        image_token_id=51289,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if isinstance(text_config, dict):
            text_config["model_type"] = text_config["model_type"] if "model_type" in text_config else "bart"
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            text_config = CONFIG_MAPPING["bart"]()

        if isinstance(vision_config, dict):
            vision_config = Florence2VisionConfig(**vision_config)
        elif vision_config is None:
            logger.info("vision_config is None. Initializing the Florence2VisionConfig with default values.")
            vision_config = Florence2VisionConfig()

        self.is_encoder_decoder = True
        self.text_config = text_config
        self.vision_config = vision_config
        self.image_token_id = image_token_id


class Florence2VisionDropPath(BeitDropPath):
    pass


class Florence2VisionLearnedAbsolutePositionEmbedding2D(DetrLearnedPositionEmbedding):
    def __init__(self, embedding_dim: int = 256, num_pos: int = 50):
        super().__init__()
        embedding_dim = embedding_dim
        self.row_embeddings = nn.Embedding(num_pos, embedding_dim // 2)
        self.column_embeddings = nn.Embedding(num_pos, embedding_dim - (embedding_dim // 2))


class Florence2VisionPositionalEmbeddingCosine1D(nn.Module):
    """
    This module generates 1D cosine positional embeddings using precomputed sinusoidal functions.
    """

    def __init__(self, embed_dim: int = 512, max_seq_len: int = 1024) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        pos_idx_to_embed = torch.empty((self.max_seq_len, self.embed_dim))
        sine, cosine = self.get_sinusoid_embeddings(
            max_positions=self.max_seq_len,
            embed_dim=self.embed_dim,
        )
        pos_idx_to_embed[:, 0::2] = sine
        pos_idx_to_embed[:, 1::2] = cosine
        # Save the positional embeddings in a constant buffer.
        self.register_buffer("pos_idx_to_embed", pos_idx_to_embed)

    @staticmethod
    def get_sinusoid_embeddings(max_positions: int, embed_dim: int):
        half_dim = embed_dim // 2
        emb = math.log(10000) / half_dim
        emb = torch.exp(torch.arange(half_dim, dtype=torch.int64).float() * -emb)
        emb = torch.arange(max_positions, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        return torch.sin(emb), torch.cos(emb)

    def forward(self, seq_embeds: torch.Tensor) -> torch.Tensor:
        len_seq = seq_embeds.size(1)
        if len_seq > self.max_seq_len:
            raise ValueError(f"Maximum sequence length {self.max_seq_len}, got {len_seq}")
        pos_embeds = self.pos_idx_to_embed[0:len_seq, :]
        return pos_embeds


class Florence2VisionMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        activation_function: str = "gelu",
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.activation_fn = ACT2FN[activation_function]
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class Florence2VisionConvEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, config: Florence2VisionConfig, stage_idx: int):
        super().__init__()
        self.config = config
        self.stage_idx = stage_idx
        self.patch_size = config.patch_size[stage_idx]
        self.in_channels = config.in_channels if stage_idx == 0 else config.embed_dim[stage_idx - 1]
        self.embed_dim = config.embed_dim[stage_idx]
        self.stride = config.patch_stride[stage_idx]
        self.padding = config.patch_padding[stage_idx]
        self.pre_norm = config.patch_prenorm[stage_idx]

        self.conv = nn.Conv2d(
            self.in_channels,
            self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.stride,
            padding=self.padding,
        )

        dim_norm = self.in_channels if self.pre_norm else self.embed_dim
        self.norm = nn.LayerNorm(dim_norm)

    def forward(self, hidden_states: torch.Tensor):
        if self.norm and self.pre_norm:
            hidden_states = hidden_states.permute(0, 2, 3, 1)
            hidden_states = self.norm(hidden_states)
            hidden_states = hidden_states.permute(0, 3, 1, 2)

        hidden_states = self.conv(hidden_states)

        if self.norm and not self.pre_norm:
            hidden_states = hidden_states.permute(0, 2, 3, 1)
            hidden_states = self.norm(hidden_states)
            hidden_states = hidden_states.permute(0, 3, 1, 2)
        return hidden_states


class Florence2VisionChannelAttention(nn.Module):
    def __init__(self, config: Florence2VisionConfig, dim: int, groups: int = 8, qkv_bias: bool = True):
        super().__init__()
        self.config = config
        self.groups = groups
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.is_causal = False

    def forward(self, hidden_states: torch.Tensor):
        batch_size, num_tokens, hidden_size = hidden_states.shape

        # Reshape for grouped channel attention
        qkv = self.qkv(hidden_states).reshape(batch_size, num_tokens, 3, self.groups, hidden_size // self.groups)
        qkv = qkv.permute(2, 0, 3, 4, 1)
        query, key, value = qkv.unbind(0)

        scale = num_tokens**-0.5
        # Channel-to-channel attention within groups:
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        hidden_states, _ = attention_interface(
            self,
            query,
            key,
            value,
            attention_mask=None,
            scaling=scale,
        )
        hidden_states = hidden_states.permute(0, 3, 2, 1)
        hidden_states = hidden_states.reshape(batch_size, num_tokens, hidden_size)

        # Final projection
        hidden_states = self.proj(hidden_states)
        return hidden_states


class Florence2VisionChannelBlock(nn.Module):
    def __init__(
        self,
        config: Florence2VisionConfig,
        stage_idx: int,
        drop_path_rate: float,
        kernel_size: int = 3,
    ):
        super().__init__()

        self.config = config
        dim_in = config.embed_dim[stage_idx]

        self.conv1 = nn.Conv2d(
            dim_in,
            dim_in,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim_in,
        )
        self.norm1 = nn.LayerNorm(config.embed_dim[stage_idx])
        self.channel_attn = Florence2VisionChannelAttention(
            config=config,
            dim=config.embed_dim[stage_idx],
            groups=config.num_groups[stage_idx],
            qkv_bias=config.qkv_bias,
        )
        self.drop_path1 = Florence2VisionDropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

        self.conv2 = nn.Conv2d(
            dim_in,
            dim_in,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim_in,
        )
        self.norm2 = nn.LayerNorm(config.embed_dim[stage_idx])
        self.ffn = Florence2VisionMLP(
            in_features=config.embed_dim[stage_idx],
            hidden_features=int(config.embed_dim[stage_idx] * config.mlp_ratio),
            activation_function=config.activation_function,
        )
        self.drop_path2 = Florence2VisionDropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def forward(self, hidden_states: torch.Tensor):
        batch_size, embed_dim, height, width = hidden_states.shape

        # First channel block: Depthwise Conv + Channel Attention
        hidden_states = self.conv1(hidden_states) + hidden_states
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        residual = hidden_states

        # Channel group attention self-attention mechanism
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.channel_attn(hidden_states)
        hidden_states = residual + self.drop_path1(hidden_states)
        hidden_states = hidden_states.transpose(1, 2).view(batch_size, embed_dim, height, width)

        # Second channel block: Depthwise Conv + FFN
        hidden_states = self.conv2(hidden_states) + hidden_states
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        residual = hidden_states

        # FFN
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = residual + self.drop_path2(hidden_states)
        hidden_states = hidden_states.transpose(1, 2).view(batch_size, embed_dim, height, width)

        return hidden_states


def window_partition(hidden_states: torch.Tensor, window_size: int):
    """Split input tensor into non-overlapping windows"""
    batch_size, height, width, embed_dim = hidden_states.shape
    hidden_states = hidden_states.view(
        batch_size, height // window_size, window_size, width // window_size, window_size, embed_dim
    )
    windows = hidden_states.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, embed_dim)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, height: int, width: int):
    """Merge windows back into original spatial layout"""
    embed_dim = windows.shape[-1]
    hidden_states = windows.view(-1, height // window_size, width // window_size, window_size, window_size, embed_dim)
    hidden_states = hidden_states.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, height, width, embed_dim)
    return hidden_states


class Florence2VisionWindowAttention(nn.Module):
    def __init__(
        self,
        config: Florence2VisionConfig,
        dim: int,
        num_heads: int,
        window_size: int,
        qkv_bias: bool = True,
    ):
        super().__init__()
        self.config = config
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.is_causal = False

    def forward(self, hidden_states: torch.Tensor):
        batch_size, height, width, embed_dim = hidden_states.shape

        pad_left = pad_top = 0
        pad_right = (self.window_size - width % self.window_size) % self.window_size
        pad_bottom = (self.window_size - height % self.window_size) % self.window_size
        # Pad the input if necessary
        hidden_states = F.pad(hidden_states, (0, 0, pad_left, pad_right, pad_top, pad_bottom))
        _, padded_height, padded_width, _ = hidden_states.shape

        # Partition input into non-overlapping windows (for local spatial attention in DaViT)
        windowed_hidden_states = window_partition(hidden_states, self.window_size)
        windowed_hidden_states = windowed_hidden_states.view(-1, self.window_size * self.window_size, embed_dim)

        # Generate Q, K, V for each window
        num_windows_per_batch, num_tokens_per_window, embed_dim = windowed_hidden_states.shape
        qkv = self.qkv(windowed_hidden_states).reshape(
            num_windows_per_batch, num_tokens_per_window, 3, self.num_heads, embed_dim // self.num_heads
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        windowed_hidden_states, _ = attention_interface(
            self,
            query,
            key,
            value,
            attention_mask=None,
            scaling=self.scale,
        )
        windowed_hidden_states = windowed_hidden_states.view(num_windows_per_batch, num_tokens_per_window, embed_dim)
        # Apply output projection
        windowed_hidden_states = self.proj(windowed_hidden_states)

        # Merge windows back to original spatial layout
        windowed_hidden_states = windowed_hidden_states.view(-1, self.window_size, self.window_size, embed_dim)
        hidden_states = window_reverse(windowed_hidden_states, self.window_size, padded_height, padded_width)

        hidden_states = hidden_states[:, :height, :width, :].contiguous()
        hidden_states = hidden_states.view(batch_size, height * width, embed_dim)

        return hidden_states


class Florence2VisionSpatialBlock(nn.Module):
    def __init__(
        self,
        config: Florence2VisionConfig,
        stage_idx: int,
        drop_path_rate: float,
        kernel_size: int = 3,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            config.embed_dim[stage_idx],
            config.embed_dim[stage_idx],
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=config.embed_dim[stage_idx],
        )
        self.norm1 = nn.LayerNorm(config.embed_dim[stage_idx])
        self.window_attn = Florence2VisionWindowAttention(
            config,
            config.embed_dim[stage_idx],
            config.num_heads[stage_idx],
            config.window_size,
            config.qkv_bias,
        )
        self.drop_path1 = Florence2VisionDropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

        self.conv2 = nn.Conv2d(
            config.embed_dim[stage_idx],
            config.embed_dim[stage_idx],
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=config.embed_dim[stage_idx],
        )
        self.norm2 = nn.LayerNorm(config.embed_dim[stage_idx])
        self.ffn = Florence2VisionMLP(
            in_features=config.embed_dim[stage_idx],
            hidden_features=int(config.embed_dim[stage_idx] * config.mlp_ratio),
            activation_function=config.activation_function,
        )
        self.drop_path2 = Florence2VisionDropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def forward(self, hidden_states: torch.Tensor):
        batch_size, embed_dim, height, width = hidden_states.shape

        # First spatial mixing block: Conv + Window Attention
        hidden_states = self.conv1(hidden_states) + hidden_states
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        residual = hidden_states

        # Spatial Window-based self-attention mechanism
        hidden_states = self.norm1(hidden_states)
        hidden_states = hidden_states.view(batch_size, height, width, embed_dim)
        hidden_states = self.window_attn(hidden_states)
        hidden_states = residual + self.drop_path1(hidden_states)
        hidden_states = hidden_states.transpose(1, 2).view(batch_size, embed_dim, height, width)

        # Second spatial mixing block: Conv + FFN
        hidden_states = self.conv2(hidden_states) + hidden_states
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        residual = hidden_states

        # FFN
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = residual + self.drop_path2(hidden_states)
        hidden_states = hidden_states.transpose(1, 2).view(batch_size, embed_dim, height, width)

        return hidden_states


class Florence2VisionBlock(nn.Module):
    def __init__(
        self,
        config: Florence2VisionConfig,
        stage_idx: int,
        spatial_drop_path_rate: float,
        channel_drop_path_rate: float,
    ):
        super().__init__()
        self.spatial_block = Florence2VisionSpatialBlock(
            config=config,
            stage_idx=stage_idx,
            drop_path_rate=spatial_drop_path_rate,
        )
        self.channel_block = Florence2VisionChannelBlock(
            config=config,
            stage_idx=stage_idx,
            drop_path_rate=channel_drop_path_rate,
        )

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.spatial_block(hidden_states)
        hidden_states = self.channel_block(hidden_states)
        return hidden_states


@auto_docstring
class Florence2VisionPreTrainedModel(PreTrainedModel):
    config_class = Florence2VisionConfig
    main_input_name = "pixel_values"
    _supports_sdpa = True
    _supports_flash_attn_2 = True

    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm, nn.BatchNorm2d]) -> None:
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(module.weight, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0)


@auto_docstring
class Florence2VisionBackbone(Florence2VisionPreTrainedModel):
    def __init__(self, config: Florence2VisionConfig):
        super().__init__(config)
        self.config = config

        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        self.num_groups = config.num_groups
        self.num_stages = len(self.embed_dim)

        if not (self.num_stages == len(self.num_heads) == len(self.num_groups)):
            raise ValueError(
                f"Expected self.num_stages ({self.num_stages}) == "
                f"len(self.num_heads) ({len(self.num_heads)}) == "
                f"len(self.num_groups) ({len(self.num_groups)})"
            )

        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(config.depths) * 2, device="cpu")]
        depth_offset = 0

        convs = []
        blocks = []
        for stage_idx in range(self.num_stages):
            conv_embed = Florence2VisionConvEmbed(
                config=config,
                stage_idx=stage_idx,
            )
            convs.append(conv_embed)

            block = nn.ModuleList(
                Florence2VisionBlock(
                    config=config,
                    stage_idx=stage_idx,
                    spatial_drop_path_rate=dpr[depth_offset + block_idx * 2],
                    channel_drop_path_rate=dpr[depth_offset + block_idx * 2 + 1],
                )
                for block_idx in range(config.depths[stage_idx])
            )
            blocks.append(block)
            depth_offset += config.depths[stage_idx] * 2

        self.convs = nn.ModuleList(convs)
        self.blocks = nn.ModuleList(blocks)

        # Initialize weights and apply final processing
        self.post_init()

    @property
    def dim_out(self):
        return self.embed_dim[-1]

    def forward(self, hidden_states: torch.Tensor):
        for conv, block in zip(self.convs, self.blocks):
            hidden_states = conv(hidden_states)
            for layer in block:
                hidden_states = layer(hidden_states)
        return hidden_states


class Florence2VisionProjector(nn.Module):
    def __init__(self, config: Florence2VisionConfig):
        super().__init__()
        self.vision_embedding_dim = config.embed_dim[-1]
        self.vision_projection_dim = config.projection_dim
        self.image_projection = nn.Linear(self.vision_embedding_dim, self.vision_projection_dim, bias=False)
        self.image_proj_norm = nn.LayerNorm(self.vision_projection_dim)
        self.image_position_embed = Florence2VisionLearnedAbsolutePositionEmbedding2D(
            embedding_dim=self.vision_embedding_dim, num_pos=config.max_position_embeddings
        )
        self.visual_temporal_embed = Florence2VisionPositionalEmbeddingCosine1D(
            embed_dim=self.vision_embedding_dim, max_seq_len=config.max_temporal_embeddings
        )

    def forward(self, image_features):
        position_features = image_features + self.image_position_embed(image_features)
        position_features = position_features.flatten(2).transpose(1, 2)
        temporal_features = self.visual_temporal_embed(position_features[:, :1, :])
        temporal_features = temporal_features.unsqueeze(1)
        visual_token_features = position_features + temporal_features
        visual_token_features = visual_token_features.unsqueeze(1)
        spatial_image_features = visual_token_features.mean(dim=2)
        temporal_image_features = visual_token_features.mean(dim=1)
        image_features = torch.cat([spatial_image_features, temporal_image_features], dim=1)
        image_features = self.image_projection(image_features)
        image_features = self.image_proj_norm(image_features)
        return image_features


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for Florence-2 base model's outputs that also contains : pre-computed hidden states that can speed up sequential
    decoding.
    """
)
class Florence2Seq2SeqModelOutput(Seq2SeqModelOutput):
    r"""
    image_hidden_states (`torch.FloatTensor`, *optional*):
    A `torch.FloatTensor` of size `(batch_size, num_image_tokens, hidden_size)`.
    image_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
    """

    image_hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for Florence-2 model's outputs that also contains : pre-computed hidden states that can speed up sequential
    decoding.
    """
)
class Florence2Seq2SeqLMOutput(Seq2SeqLMOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Language modeling loss (for next-token prediction).
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
        Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    image_hidden_states (`torch.FloatTensor`, *optional*):
        A `torch.FloatTensor` of size `(batch_size, num_image_tokens, hidden_size)`.
        image_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
    """

    image_hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None


@auto_docstring
class Florence2PreTrainedModel(LlavaPreTrainedModel):
    config_class = Florence2Config

    def _init_weights(self, module):
        std = self.config.vision_config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()


@auto_docstring(
    custom_intro="""
    Florence-2 is an advanced vision foundation model that uses a prompt-based approach to handle a wide range of vision and vision-language tasks. Florence-2 can interpret simple text prompts to perform tasks like captioning, object detection, and segmentation.
    """
)
class Florence2Model(Florence2PreTrainedModel):
    _tied_weights_keys = [
        "language_model.encoder.embed_tokens.weight",
        "language_model.decoder.embed_tokens.weight",
    ]

    def __init__(self, config: Florence2Config):
        super().__init__(config)
        self.config = config
        self.vocab_size = config.text_config.vocab_size

        self.vision_tower = Florence2VisionBackbone(config=config.vision_config)
        self.vision_projector = Florence2VisionProjector(config=config.vision_config)
        self.language_model = AutoModel.from_config(config=config.text_config)

        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self.post_init()

    def get_encoder(self):
        return self.language_model.get_encoder()

    def get_decoder(self):
        return self.language_model.get_decoder()

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.language_model.set_input_embeddings(value)

    def get_input_embeddings(self) -> nn.Module:
        return self.language_model.get_input_embeddings()

    @auto_docstring
    def get_image_features(self, pixel_values: torch.Tensor, **kwargs):
        image_features = self.vision_tower(pixel_values, **kwargs)
        image_embeds = self.vision_projector(image_features)
        return image_embeds

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_outputs: Optional[list[torch.FloatTensor]] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[tuple, Florence2Seq2SeqModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            if (input_ids is None) ^ (inputs_embeds is not None):
                raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

            if inputs_embeds is None:
                inputs_embeds = self.get_input_embeddings()(input_ids)

            if pixel_values is not None:
                image_features = self.get_image_features(pixel_values)

                if input_ids is None:
                    special_image_mask = inputs_embeds == self.get_input_embeddings()(
                        torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
                    )
                    special_image_mask = special_image_mask.all(-1)
                else:
                    special_image_mask = input_ids == self.config.image_token_id

                n_image_tokens = (special_image_mask).sum()
                special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)

                if (
                    not is_torchdynamo_compiling()
                    and inputs_embeds[special_image_mask].numel() != image_features.numel()
                ):
                    n_image_features = image_features.shape[0] * image_features.shape[1]
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )

                image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

            encoder_outputs = self.language_model.encoder(
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )

        if decoder_input_ids is None:
            decoder_start_token_id = self.config.text_config.decoder_start_token_id
            decoder_input_ids = torch.ones((inputs_embeds.size()[0], 1), dtype=torch.long, device=inputs_embeds.device)
            decoder_input_ids *= decoder_start_token_id

        decoder_outputs = self.language_model.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            return_dict=True,
            **kwargs,
        )

        return Florence2Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
        )


@auto_docstring(
    custom_intro="""
    Florence-2 is an advanced vision foundation model that uses a prompt-based approach to handle a wide range of vision and vision-language tasks. Florence-2 can interpret simple text prompts to perform tasks like captioning, object detection, and segmentation.
    """
)
class Florence2ForConditionalGeneration(Florence2PreTrainedModel, GenerationMixin):
    _tied_weights_keys = [
        "model.language_model.shared.weight",
        "model.language_model.encoder.embed_tokens.weight",
        "model.language_model.decoder.embed_tokens.weight",
    ]

    def __init__(self, config: Florence2Config):
        super().__init__(config)
        self.model = Florence2Model(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def get_input_embeddings(self) -> nn.Module:
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    @auto_docstring
    def get_image_features(self, pixel_values: torch.Tensor, **kwargs):
        return self.model.get_image_features(pixel_values=pixel_values, **kwargs)

    @property
    def language_model(self):
        return self.model.language_model

    @property
    def vision_tower(self):
        return self.model.vision_tower

    @property
    def vision_projector(self):
        return self.model.vision_projector

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[list[torch.FloatTensor]] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Florence2Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Florence2ForConditionalGeneration

        >>> model = Florence2ForConditionalGeneration.from_pretrained("microsoft/Florence-2-large")
        >>> processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large")

        >>> prompt = "<CAPTION>"
        >>> url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(text=prompt, images=image, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(**inputs, max_length=100)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "A green car parked in front of a yellow building."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
            )

        return Florence2Seq2SeqLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            image_hidden_states=outputs.image_hidden_states,
        )

    def _prepare_encoder_decoder_kwargs_for_generation(
        self,
        inputs_tensor: torch.Tensor,
        model_kwargs,
        model_input_name: Optional[str],
        generation_config,
    ) -> dict[str, Any]:
        # override to handle merging image and text embeddings before passing to language encoder
        inputs_embeds = model_kwargs.pop("inputs_embeds", None)
        pixel_values = model_kwargs.pop("pixel_values", None)

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(inputs_tensor)

        if pixel_values is not None:
            image_features = self.get_image_features(pixel_values)

            if inputs_tensor is None:
                special_image_mask = inputs_embeds == self.get_input_embeddings()(
                    torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
                special_image_mask = special_image_mask.all(-1)
            else:
                special_image_mask = inputs_tensor == self.config.image_token_id

            n_image_tokens = (special_image_mask).sum()
            special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)

            if not is_torchdynamo_compiling() and inputs_embeds[special_image_mask].numel() != image_features.numel():
                n_image_features = image_features.shape[0] * image_features.shape[1]
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        model_kwargs["inputs_embeds"] = inputs_embeds
        return super()._prepare_encoder_decoder_kwargs_for_generation(
            None, model_kwargs, model_input_name, generation_config
        )


__all__ = [
    "Florence2Config",
    "Florence2VisionConfig",
    "Florence2Model",
    "Florence2ForConditionalGeneration",
    "Florence2PreTrainedModel",
    "Florence2VisionBackbone",
    "Florence2VisionPreTrainedModel",
]
