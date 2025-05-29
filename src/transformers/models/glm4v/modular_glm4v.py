# coding=utf-8
# Copyright 2025 The GLM4 & ZhipuAI team and HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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

import itertools
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn import LayerNorm

from transformers.models.glm4.modeling_glm4 import Glm4MLP, Glm4RMSNorm
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig, Qwen2_5_VLTextConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionPatchEmbed,
    Qwen2_5_VisionRotaryEmbedding,
    Qwen2_5_VLDecoderLayer,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLMLP,
    Qwen2_5_VLModel,
    Qwen2_5_VLPreTrainedModel,
    Qwen2_5_VLTextModel,
    Qwen2_5_VLVisionAttention,
    Qwen2_5_VLVisionBlock,
    Qwen2_5_VLVisionFlashAttention2,
    Qwen2_5_VLVisionSdpaAttention,
)
from transformers.models.qwen2_5_vl.processing_qwen2_5_vl import (
    Qwen2_5_VLImagesKwargs,
    Qwen2_5_VLProcessor,
    Qwen2_5_VLProcessorKwargs,
    Qwen2_5_VLVideosProcessorKwargs,
)

from ...activations import ACT2FN
from ...configuration_utils import PretrainedConfig
from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import logging
from ...video_utils import VideoInput


logger = logging.get_logger(__name__)


class Glm4VisionMlp(Qwen2_5_VLMLP):
    def __init__(self, config, bias: bool = False):
        super().__init__(config, bias)
        self.intermediate_size = config.out_hidden_size


class Glm4vMLP(Glm4MLP):
    pass


class Glm4vPatchMerger(nn.Module):
    def __init__(self, dim: int, context_dim: int, hidden_act: str, bias: bool = False) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=bias)
        self.post_projection_norm = LayerNorm(dim)
        self.gate_proj = nn.Linear(dim, context_dim, bias=bias)
        self.up_proj = nn.Linear(dim, context_dim, bias=bias)
        self.down_proj = nn.Linear(context_dim, dim, bias=bias)
        self.act1 = nn.GELU()
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.proj(hidden_state)
        hidden_state = self.act1(self.post_projection_norm(hidden_state))
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


class Glm4vVisionConfig(PretrainedConfig):
    model_type = "glm4v"
    base_config_key = "vision_config"

    def __init__(
        self,
        depth=24,
        hidden_size=1536,
        hidden_act="silu",
        num_heads=12,
        in_channels=3,
        image_size=336,
        patch_size=14,
        rms_norm_eps=1e-05,
        spatial_merge_size=2,
        temporal_patch_size=1,
        out_hidden_size=4096,
        intermediate_size=13696,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.depth = depth
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.out_hidden_size = out_hidden_size
        self.intermediate_size = intermediate_size
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps


class Glm4vTextConfig(Qwen2_5_VLTextConfig):
    r"""
    This is the configuration class to store the configuration of a [`Glm4vModel`]. It is used to instantiate a
    GLM-4V-0414 model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of
    GLM-4V-9B-0414 [THUDM/GLM-4V-9B-0414](https://huggingface.co/THUDM/GLM-4V-9B-0414).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 151552):
            Vocabulary size of the Glm4v model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`Glm4vModel`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 13696):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 40):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 2):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to `32`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 32768):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        use_sliding_window (`bool`, *optional*, defaults to `False`):
            Whether to use sliding window attention.
        sliding_window (`int`, *optional*, defaults to 4096):
            Sliding window attention (SWA) window size. If not specified, will default to `4096`.
        max_window_layers (`int`, *optional*, defaults to 40):
            The number of layers that use SWA (Sliding Window Attention). The bottom layers use SWA while the top use full attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. NOTE: if you apply new rope type
            and you expect the model to work on longer `max_position_embeddings`, we recommend you to update this value
            accordingly.
            Expected contents:
                `rope_type` (`str`):
                    The sub-variant of RoPE to use. Can be one of ['default', 'linear', 'dynamic', 'yarn', 'longrope',
                    'llama3'], with 'default' being the original RoPE implementation.
                `factor` (`float`, *optional*):
                    Used with all rope types except 'default'. The scaling factor to apply to the RoPE embeddings. In
                    most scaling types, a `factor` of x will enable the model to handle sequences of length x *
                    original maximum pre-trained length.
                `original_max_position_embeddings` (`int`, *optional*):
                    Used with 'dynamic', 'longrope' and 'llama3'. The original max position embeddings used during
                    pretraining.
                `attention_factor` (`float`, *optional*):
                    Used with 'yarn' and 'longrope'. The scaling factor to be applied on the attention
                    computation. If unspecified, it defaults to value recommended by the implementation, using the
                    `factor` field to infer the suggested value.
        image_token_id (`int`, *optional*):
            Token index used as placeholder for image embeddings.
        video_token_id (`int`, *optional*):
            Token index used as placeholder for video embeddings.

    ```python
    >>> from transformers import Glm4vTextModel, Glm4vConfig

    >>> # Initializing a GLM-4V-0414 style configuration
    >>> configuration = Glm4vConfig()

    >>> # Initializing a model from the GLM-4V-0414 style configuration
    >>> model = Glm4vTextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "glm4v_text"

    def __init__(
        self,
        vocab_size=151552,
        hidden_size=4096,
        intermediate_size=13696,
        num_hidden_layers=40,
        num_attention_heads=32,
        num_key_value_heads=2,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-05,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=40,
        attention_dropout=0.0,
        rope_scaling=None,
        image_token_id=None,
        video_token_id=None,
        **kwargs,
    ):
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


class Glm4vConfig(Qwen2_5_VLConfig):
    r"""
    This is the configuration class to store the configuration of a [`Glm4vModel`]. It is used to instantiate a
    GLM-4V-0414 model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of
    GLM-4V-9B-0414 [THUDM/GLM-4V-9B-0414](https://huggingface.co/THUDM/GLM-4V-9B-0414).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        text_config (`Union[PreTrainedConfig, dict]`, *optional*, defaults to `Glm4vTextConfig`):
            The config object or dictionary of the text backbone.
        vision_config (`Union[PreTrainedConfig, dict]`,  *optional*, defaults to `Glm4vVisionConfig`):
            The config object or dictionary of the vision backbone.
        image_token_id (`int`, *optional*, defaults to 151343):
            The image token index to encode the image prompt.
        video_token_id (`int`, *optional*, defaults to 151344):
            The video token index to encode the image prompt.
        vision_start_token_id (`int`, *optional*, defaults to 151341):
            The vision start token index to encode the start of vision.
        vision_end_token_id (`int`, *optional*, defaults to 151342):
            The vision end token index to encode the end of vision.

    ```python
    >>> from transformers import Glm4vForConditionalGeneration, Glm4vConfig

    >>> # Initializing a GLM-4V-0414 style configuration
    >>> configuration = Glm4vConfig()

    >>> # Initializing a model from the GLM-4V-9B-0414 style configuration
    >>> model = Glm4vForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "glm4v"
    sub_configs = {"vision_config": Glm4vVisionConfig, "text_config": Glm4vTextConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        image_token_id=151343,
        video_token_id=151344,
        vision_start_token_id=151341,
        vision_end_token_id=151342,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id


class Glm4vVisionPatchEmbed(Qwen2_5_VisionPatchEmbed):
    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 1,
        in_channels: int = 3,
        embed_dim: int = 1536,
        bias: bool = True,
    ) -> None:
        super().__init__(
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        kernel_size = [temporal_patch_size, patch_size, patch_size]
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=bias)


class Glm4vVisionRotaryEmbedding(Qwen2_5_VisionRotaryEmbedding):
    pass


class Glm4vRMSNorm(Glm4RMSNorm):
    pass


class Glm4vVisionEmbeddings(nn.Module):
    def __init__(self, config: Glm4vVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False)

    def adapt_position_encoding(
        self, lengths: torch.Tensor, image_shapes: torch.Tensor, h_coords: torch.Tensor, w_coords: torch.Tensor
    ):
        """
        Using 2D interpolation to adapt position encoding

        Args:
            lengths: List or Tensor of sequence lengths for each image [batch_size]
            image_shapes: Image shape Tensor [batch_size, 3] (t h w) where index 1 is H, index 2 is W
            h_coords: h coordinate for each patch Tensor [total_seq]
            w_coords: w coordinate for each patch Tensor [total_seq]

        Returns:
            adapted_pos_embed: Adapted position encoding Tensor [total_seq, hidden_size]
        """
        pos_embed_weight = self.position_embedding.weight
        hidden_size = pos_embed_weight.shape[1]
        total_seq = h_coords.shape[0]
        device = pos_embed_weight.device
        h_coords, w_coords = h_coords.to(device), w_coords.to(device)
        if total_seq == 0:
            return torch.empty(0, hidden_size, device=device, dtype=pos_embed_weight.dtype)

        if isinstance(lengths, list):
            lengths = torch.tensor(lengths, device=device, dtype=torch.long)
        if not isinstance(image_shapes, torch.Tensor):
            image_shapes = torch.tensor(image_shapes, device=device, dtype=torch.long)

        orig_size_sq = pos_embed_weight.shape[0]
        orig_size = int(orig_size_sq**0.5)
        pos_embed_2d = pos_embed_weight.view(orig_size, orig_size, hidden_size)
        pos_embed_2d = pos_embed_2d.permute(2, 0, 1).unsqueeze(0)
        target_h_list = [image_shapes[i, 1].repeat(lengths[i]) for i in range(len(lengths))]
        target_w_list = [image_shapes[i, 2].repeat(lengths[i]) for i in range(len(lengths))]
        target_h = torch.cat(target_h_list)
        target_w = torch.cat(target_w_list)
        h_coords_float = h_coords.float()
        w_coords_float = w_coords.float()
        target_h_float = target_h.float()
        target_w_float = target_w.float()
        norm_w = ((w_coords_float + 0.5) / target_w_float) * 2 - 1
        norm_h = ((h_coords_float + 0.5) / target_h_float) * 2 - 1
        grid = torch.stack((norm_w, norm_h), dim=-1)
        grid = grid.unsqueeze(0).unsqueeze(2)
        pos_embed_2d_fp32 = pos_embed_2d.float()
        grid_fp32 = grid.float()
        interpolated_embed_fp32 = F.grid_sample(
            pos_embed_2d_fp32, grid_fp32, mode="bicubic", align_corners=False, padding_mode="border"
        )
        adapted_pos_embed_fp32 = interpolated_embed_fp32.squeeze(0).squeeze(-1).permute(1, 0)
        adapted_pos_embed = adapted_pos_embed_fp32.to(pos_embed_weight.dtype)
        return adapted_pos_embed

    def forward(self, embeddings, lengths, image_shapes, h_coords, w_coords) -> torch.Tensor:
        embeddings = embeddings + self.adapt_position_encoding(lengths, image_shapes, h_coords, w_coords)
        return embeddings


def rotate_half_llm(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    """Applies Rotary Position Embedding with Multimodal Sections to the query and key tensors (https://qwenlm.github.io/blog/qwen2-vl/).

    Explanation:
        Multimodal 3D rotary position embedding is an extension to 1D rotary position embedding. The input embedding
        sequence contains vision (images / videos) embedding and text embedding or just contains text embedding. For
        vision embedding part, we apply rotary position embedding on temporal, height and width dimension separately.
        Here we split the channel dimension to 3 chunks for the temporal, height and width rotary position embedding.
        For text embedding part, we just apply 1D rotary position embedding. The three rotary position index (temporal,
        height and width) of text embedding is always the same, so the text embedding rotary position embedding has no
        difference with modern LLMs.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        mrope_section(`List(int)`):
            Multimodal rope section is for channel dimension of temporal, height and width in rope calculation.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    mrope_section = mrope_section * 2
    cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )
    sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )

    # Interleave them instead of usual shape
    cos = cos[..., : cos.shape[-1] // 2].repeat_interleave(2, dim=-1)
    sin = sin[..., : sin.shape[-1] // 2].repeat_interleave(2, dim=-1)

    # Keep half or full tensor for later concatenation
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    # Apply rotary embeddings on the first half or full tensor
    q_embed = (q_rot * cos) + (rotate_half_llm(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half_llm(k_rot) * sin)

    # Concatenate back to full shape
    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)

    return q_embed, k_embed


class Glm4vVisionFlashAttention2(Qwen2_5_VLVisionFlashAttention2):
    def __init__(self, dim: int, num_heads: int = 12) -> None:
        super().__init__(dim=dim, num_heads=num_heads)
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)


class Glm4vVisionAttention(Qwen2_5_VLVisionAttention):
    def __init__(self, dim: int, num_heads: int = 12) -> None:
        super().__init__(dim=dim, num_heads=num_heads)
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)


class Glm4vVisionSdpaAttention(Qwen2_5_VLVisionSdpaAttention):
    def __init__(self, dim: int, num_heads: int = 12) -> None:
        super().__init__(dim=dim, num_heads=num_heads)
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)


GLM4V_VISION_ATTENTION_CLASSES = {
    "eager": Glm4vVisionAttention,
    "flash_attention_2": Glm4vVisionFlashAttention2,
    "sdpa": Glm4vVisionSdpaAttention,
}


class Glm4vVisionBlock(Qwen2_5_VLVisionBlock):
    def __init__(self, config, attn_implementation: str = "sdpa") -> None:
        super().__init__()
        self.norm1 = Glm4vRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm2 = Glm4vRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = GLM4V_VISION_ATTENTION_CLASSES[attn_implementation](config.hidden_size, num_heads=config.num_heads)
        self.mlp = Glm4VisionMlp(config, bias=False)


class Glm4vPreTrainedModel(Qwen2_5_VLPreTrainedModel):
    config_class = Glm4vConfig
    _no_split_modules = ["Glm4vDecoderLayer", "Glm4vVisionBlock"]
    _skip_keys_device_placement = "past_key_values"

    def _init_weights(self, module):
        std = self.config.get_text_config().initializer_range
        if isinstance(module, (nn.Linear, nn.Conv3d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, Glm4vRMSNorm):
            module.weight.data.fill_(1.0)


class Glm4vVisionTransformerPretrainedModel(Glm4vPreTrainedModel):
    config_class = Glm4vVisionConfig
    _no_split_modules = ["Glm4vVisionBlock"]

    def __init__(self, config, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size

        self.embeddings = Glm4vVisionEmbeddings(config)
        self.patch_embed = Glm4vVisionPatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            embed_dim=config.hidden_size,
            bias=True,
        )

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Glm4vVisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList(
            [Glm4vVisionBlock(config, config._attn_implementation) for _ in range(config.depth)]
        )
        self.merger = Glm4vPatchMerger(
            dim=config.out_hidden_size, context_dim=config.intermediate_size, hidden_act=config.hidden_act
        )

        self.post_conv_layernorm = Glm4vRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.downsample = nn.Conv2d(
            in_channels=config.hidden_size,
            out_channels=config.out_hidden_size,
            kernel_size=config.spatial_merge_size,
            stride=config.spatial_merge_size,
        )
        self.post_layernorm = Glm4vRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False

    def rot_pos_emb(self, grid_thw):
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb, pos_ids

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
                The final hidden states of the model.
            grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
                The temporal, height and width of feature shape of each image in LLM.

        Returns:
            `torch.Tensor`: hidden_states.
        """
        hidden_states = self.patch_embed(hidden_states)
        hidden_states = self.post_conv_layernorm(hidden_states)

        rotary_pos_emb, image_type_ids = self.rot_pos_emb(grid_thw)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
            # See https://github.com/huggingface/transformers/pull/34852 for more information
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        hidden_states = self.embeddings(hidden_states, seqlens, grid_thw, image_type_ids[:, 0], image_type_ids[:, 1])

        for blk in self.blocks:
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    blk.__call__, hidden_states, cu_seqlens, None, position_embeddings
                )
            else:
                hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens, position_embeddings=position_embeddings)

        hidden_states = self.post_layernorm(hidden_states)

        hidden_states = hidden_states.view(
            -1, self.spatial_merge_size, self.spatial_merge_size, hidden_states.shape[-1]
        )
        hidden_states = hidden_states.permute(0, 3, 1, 2)
        hidden_states = self.downsample(hidden_states).view(-1, self.config.out_hidden_size)

        hidden_states = self.merger(hidden_states)
        return hidden_states


class Glm4vDecoderLayer(Qwen2_5_VLDecoderLayer):
    def __init__(self, config: Glm4vTextConfig, layer_idx: int):
        super().__init__()
        self.mlp = Glm4vMLP(config)
        self.input_layernorm = Glm4vRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Glm4vRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_self_attn_layernorm = Glm4vRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_mlp_layernorm = Glm4vRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )

        hidden_states = self.post_self_attn_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_mlp_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class Glm4vModel(Qwen2_5_VLModel):
    config_class = Glm4vConfig
    _checkpoint_conversion_mapping = None

    def __init__(self, config):
        super().__init__(config)
        self.visual = Glm4vVisionTransformerPretrainedModel._from_config(config.vision_config)

    def get_rope_index(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

        Explanation:
            Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

            For pure text embedding sequence, the rotary position embedding has no difference with modern LLMs.
            Examples:
                input_ids: [T T T T T], here T is for text.
                temporal position_ids: [0, 1, 2, 3, 4]
                height position_ids: [0, 1, 2, 3, 4]
                width position_ids: [0, 1, 2, 3, 4]

            For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
            and 1D rotary position embedding for text part.
            Examples:
                Temporal (Time): 3 patches, representing different segments of the video in time.
                Height: 2 patches, dividing each frame vertically.
                Width: 2 patches, dividing each frame horizontally.
                We also have some important parameters:
                fps (Frames Per Second): The video's frame rate, set to 1. This means one frame is processed each second.
                tokens_per_second: This is a crucial parameter. It dictates how many "time-steps" or "temporal tokens" are conceptually packed into a one-second interval of the video. In this case, we have 25 tokens per second. So each second of the video will be represented with 25 separate time points. It essentially defines the temporal granularity.
                temporal_patch_size: The number of frames that compose one temporal patch. Here, it's 2 frames.
                interval: The step size for the temporal position IDs, calculated as tokens_per_second * temporal_patch_size / fps. In this case, 25 * 2 / 1 = 50. This means that each temporal patch will be have a difference of 50 in the temporal position IDs.
                input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
                vision temporal position_ids: [0, 0, 0, 0, 50, 50, 50, 50, 100, 100, 100, 100]
                vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
                vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                text temporal position_ids: [101, 102, 103, 104, 105]
                text height position_ids: [101, 102, 103, 104, 105]
                text width position_ids: [101, 102, 103, 104, 105]
                Here we calculate the text start position_ids as the max vision position_ids plus 1.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
                it.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
            second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
                The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

        Returns:
            position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
            mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
        """

        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        vision_start_token_id = self.config.vision_start_token_id
        vision_end_token_id = self.config.vision_end_token_id

        mrope_position_deltas = []
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )

            attention_mask = attention_mask.to(total_input_ids.device)
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]
                input_tokens = input_ids.tolist()

                input_token_type = []
                video_check_flg = False
                for token in input_tokens:
                    if token == vision_start_token_id:
                        video_check_flg = True
                    elif token == vision_end_token_id:
                        video_check_flg = False

                    if token == image_token_id and not video_check_flg:
                        input_token_type.append("image")
                    elif token == image_token_id and video_check_flg:
                        input_token_type.append("video")
                    else:
                        input_token_type.append("text")

                input_type_group = []
                for key, group in itertools.groupby(enumerate(input_token_type), lambda x: x[1]):
                    group = list(group)
                    start_index = group[0][0]
                    end_index = group[-1][0] + 1
                    input_type_group.append((key, start_index, end_index))

                llm_pos_ids_list = []
                video_frame_num = 1
                image_index, video_index = 0, 0

                for modality_type, start_idx, end_idx in input_type_group:
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0

                    if modality_type == "image":
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        llm_grid_t, llm_grid_h, llm_grid_w = (
                            t.item(),
                            h.item() // spatial_merge_size,
                            w.item() // spatial_merge_size,
                        )

                        t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                        h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                        w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                        llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + st_idx)

                        image_index += 1
                        video_frame_num = 1

                    elif modality_type == "video":
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        second_per_grid_t = second_per_grid_ts[video_index] if second_per_grid_ts is not None else 1.0
                        llm_grid_t, llm_grid_h, llm_grid_w = (
                            t.item(),
                            h.item() // spatial_merge_size,
                            w.item() // spatial_merge_size,
                        )

                        range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                        expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)

                        second_per_grid_t = torch.as_tensor(
                            second_per_grid_t, dtype=range_tensor.dtype, device=range_tensor.device
                        )

                        time_tensor = expanded_range * second_per_grid_t * self.config.vision_config.tokens_per_second
                        time_tensor_long = time_tensor.long()
                        t_index = time_tensor_long.flatten()

                        h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                        w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                        llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + st_idx)

                        video_index += 1
                        video_frame_num += 1

                    else:
                        text_len = end_idx - start_idx
                        llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                        video_frame_num = 1

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )

            return position_ids, mrope_position_deltas


class Glm4vTextModel(Qwen2_5_VLTextModel):
    def __init__(self, config: Glm4vTextConfig):
        super().__init__(config)
        self.norm = Glm4vRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class Glm4vForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    _checkpoint_conversion_mapping = None
    config_class = Glm4vConfig


class Glm4vVideosProcessorKwargs(Qwen2_5_VLVideosProcessorKwargs):
    pass


class Glm4vImagesKwargs(Qwen2_5_VLImagesKwargs):
    pass


class Glm4vProcessorKwargs(Qwen2_5_VLProcessorKwargs):
    images_kwargs: Glm4vImagesKwargs
    videos_kwargs: Glm4vVideosProcessorKwargs
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
        "videos_kwargs": {"fps": 2.0},
    }


class Glm4vProcessor(Qwen2_5_VLProcessor):
    r"""
    Constructs a GLM-4V processor which wraps a GLM-4V image processor and a GLM-4 tokenizer into a single processor.
    [`~Glm4vProcessor.__call__`] and [`~Glm4vProcessor.decode`] for more information.
    Args:
        image_processor ([`Glm4vProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`PreTrainedTokenizerFast`], *optional*):
            The tokenizer is a required input.
        video_processor ([`Glm4vVideoProcessor`], *optional*):
            The video processor is a required input.
        chat_template (`str`, *optional*): A Jinja template which will be used to convert lists of messages
            in a chat into a tokenizable string.
    """

    tokenizer_class = ("PreTrainedTokenizer", "PreTrainedTokenizerFast")

    def __init__(self, image_processor=None, tokenizer=None, video_processor=None, chat_template=None, **kwargs):
        super().__init__(image_processor, tokenizer, video_processor, chat_template=chat_template)
        self.image_token = "<|image|>" if not hasattr(tokenizer, "image_token") else tokenizer.image_token
        self.video_token = "<|video|>" if not hasattr(tokenizer, "video_token") else tokenizer.video_token

    def __call__(
        self,
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        videos: VideoInput = None,
        **kwargs: Unpack[Glm4vProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to PreTrainedTokenizerFast's [`~PreTrainedTokenizerFast.__call__`] if `text` is not `None` to encode
        the text.

        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            videos (`np.ndarray`, `torch.Tensor`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of videos to be prepared. Each video can be a 4D NumPy array or PyTorch
                tensor, or a nested list of 3D frames. Both channels-first and channels-last formats are supported.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:
                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
            - **pixel_values_videos** -- Pixel values of videos to be fed to a model. Returned when `videos` is not `None`.
            - **image_grid_thw** -- List of image 3D grid in LLM. Returned when `images` is not `None`.
            - **video_grid_thw** -- List of video 3D grid in LLM. Returned when `videos` is not `None`.
            - **second_per_grid_ts** -- List of video seconds per time grid. Returned when `videos` is not `None`.
        """
        output_kwargs = self._merge_kwargs(
            Glm4vProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        if images is not None:
            image_inputs = self.image_processor(images=images, videos=None, **output_kwargs["images_kwargs"])
            image_grid_thw = image_inputs["image_grid_thw"]
        else:
            image_inputs = {}
            image_grid_thw = None

        if videos is not None:
            videos_inputs = self.image_processor(images=None, videos=videos, **output_kwargs["images_kwargs"])
            video_grid_thw = videos_inputs["video_grid_thw"]

            fps = output_kwargs["videos_kwargs"].pop("fps", 2.0)
            if isinstance(fps, (int, float)):
                second_per_grid_ts = [self.image_processor.temporal_patch_size / fps] * len(video_grid_thw)
            elif hasattr(fps, "__len__") and len(fps) == len(video_grid_thw):
                second_per_grid_ts = [self.image_processor.temporal_patch_size / tmp for tmp in fps]
            else:
                raise ValueError(
                    f"The length of fps ({len(fps) if hasattr(fps, '__len__') else fps}) must be equal to the length of video_grid_thw ({len(video_grid_thw)}) or fps should be a single number."
                )
            videos_inputs.update({"second_per_grid_ts": second_per_grid_ts})

        else:
            videos_inputs = {}
            video_grid_thw = None

        if not isinstance(text, list):
            text = [text]

        text = text.copy()  # below lines change text in-place
        if image_grid_thw is not None:
            merge_length = self.image_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while self.image_token in text[i]:
                    num_image_tokens = image_grid_thw[index].prod() // merge_length
                    text[i] = text[i].replace(self.image_token, "<|placeholder|>" * num_image_tokens, 1)
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.image_token)

        if video_grid_thw is not None:
            merge_length = self.image_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while self.video_token in text[i]:
                    text[i] = text[i].replace(
                        self.video_token, "<|placeholder|>" * (video_grid_thw[index].prod() // merge_length), 1
                    )
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.video_token)

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
        self._check_special_mm_tokens(text, text_inputs, modalities=["image", "video"])

        return BatchFeature(data={**text_inputs, **image_inputs, **videos_inputs}, tensor_type=return_tensors)


__all__ = [
    "Glm4vConfig",
    "Glm4vTextConfig",
    "Glm4vForConditionalGeneration",
    "Glm4vModel",
    "Glm4vPreTrainedModel",
    "Glm4vProcessor",
    "Glm4vTextModel",  # noqa: F822
]
