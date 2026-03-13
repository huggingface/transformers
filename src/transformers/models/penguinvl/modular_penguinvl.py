# Copyright 2025 Tencent and The HuggingFace Team. All rights reserved.
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
"""PyTorch PenguinVL model."""

import copy
import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from ...cache_utils import Cache
from ...configuration_utils import PreTrainedConfig
from ...feature_extraction_utils import BatchFeature
from ...generation import GenerationMixin
from ...image_transforms import convert_to_rgb, resize
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    SizeDict,
    get_image_size,
    infer_channel_dimension_format,
    is_valid_image,
    load_image,
    to_numpy_array,
    validate_preprocess_arguments,
)
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, ModelOutput
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS, RopeParameters, dynamic_rope_update
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import (
    TensorType,
    auto_docstring,
    can_return_tuple,
    is_av_available,
    is_cv2_available,
    is_decord_available,
    is_torchcodec_available,
    is_torchvision_available,
    is_vision_available,
    logging,
)
from ...utils.generic import is_flash_attention_requested
from ...utils.output_capturing import capture_outputs
from ..qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor, Qwen2VLImageProcessorKwargs, smart_resize
from ..qwen2_vl.image_processing_qwen2_vl_fast import Qwen2VLImageProcessorFast
from ..qwen3.configuration_qwen3 import Qwen3Config
from ..qwen3.modeling_qwen3 import Qwen3MLP, Qwen3Model, Qwen3RMSNorm, eager_attention_forward, rotate_half


if is_vision_available():
    from PIL import Image


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="tencent/Penguin-VL-8B")
class PenguinVLVisionConfig(PreTrainedConfig):
    r"""
    Configuration for the PenguinVL vision encoder.

    Args:
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimensionality of the encoder hidden states.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (feed-forward) layer in the Transformer encoder.
        num_hidden_layers (`int`, *optional*, defaults to 28):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            Number of key-value heads for grouped-query attention.
        head_dim (`int`, *optional*, defaults to 128):
            Dimension of each attention head.
        num_channels (`int`, *optional*, defaults to 3):
            Number of input channels.
        patch_size (`int`, *optional*, defaults to 14):
            The size of each image patch.
        hidden_act (`str`, *optional*, defaults to `"silu"`):
            The non-linear activation function in the encoder.
        rms_norm_eps (`float`, *optional*, defaults to 1e-6):
            The epsilon used by the rms normalization layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in attention layers.
        rope_theta (`float`, *optional*, defaults to 1000000.0):
            The base period of the RoPE embeddings.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated normal initializer.
    """

    model_type = "penguinvl_vision"
    base_config_key = "vision_encoder_config"

    def __init__(
        self,
        hidden_size=1024,
        intermediate_size=3072,
        num_hidden_layers=28,
        num_attention_heads=16,
        num_key_value_heads=8,
        max_position_embeddings=40960,
        rope_parameters: RopeParameters | dict[str, RopeParameters] | None = None,
        head_dim=128,
        num_channels=3,
        patch_size=14,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        attention_dropout=0.0,
        attention_bias=False,
        rope_scaling=None,
        rope_theta=1000000.0,
        initializer_range=0.02,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.head_dim = head_dim
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.hidden_act = hidden_act
        self.rms_norm_eps = rms_norm_eps
        self.attention_dropout = attention_dropout
        self.attention_bias = attention_bias
        self.rope_scaling = rope_scaling
        self.rope_theta = rope_theta
        self.initializer_range = initializer_range
        if rope_parameters is None:
            rope_parameters = {"rope_type": "default", "rope_theta": rope_theta}
        self.rope_parameters = rope_parameters

        super().__init__(**kwargs)


@auto_docstring(checkpoint="tencent/Penguin-VL-8B")
class PenguinVLConfig(Qwen3Config):
    r"""
    Configuration for the PenguinVL model.

    Args:
        vision_encoder_config (`PenguinVLVisionConfig` or `dict`, *optional*):
            Configuration for the vision encoder.
        image_token_id (`int`, *optional*, defaults to 151669):
            Token ID for the image placeholder token.
        vision_projector_type (`str`, *optional*, defaults to `"mlp2x_gelu"`):
            Type of the vision projector.
        tie_word_embeddings (`bool`, *optional*, defaults to `True`):
            Whether to tie word embeddings.
    """

    model_type = "penguinvl"
    sub_configs = {"vision_encoder_config": PenguinVLVisionConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vision_encoder_config=None,
        image_token_id=151669,
        vision_projector_type="mlp2x_gelu",
        vocab_size: int | None = 151936,
        hidden_size: int | None = 4096,
        intermediate_size: int | None = 22016,
        num_hidden_layers: int | None = 32,
        num_attention_heads: int | None = 32,
        num_key_value_heads: int | None = 32,
        head_dim: int | None = 128,
        hidden_act: str | None = "silu",
        max_position_embeddings: int | None = 32768,
        initializer_range: float | None = 0.02,
        rms_norm_eps: float | None = 1e-6,
        use_cache: bool | None = True,
        tie_word_embeddings: bool | None = False,
        rope_parameters: RopeParameters | dict[str, RopeParameters] | None = None,
        attention_bias: bool | None = False,
        use_sliding_window: bool | None = False,
        sliding_window: int | None = 4096,
        max_window_layers: int | None = 28,
        layer_types: list[str] | None = None,
        attention_dropout: float | None = 0.0,
        pad_token_id: int | None = None,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            rms_norm_eps=rms_norm_eps,
            use_cache=use_cache,
            tie_word_embeddings=tie_word_embeddings,
            rope_parameters=rope_parameters,
            attention_bias=attention_bias,
            use_sliding_window=use_sliding_window,
            sliding_window=sliding_window,
            max_window_layers=max_window_layers,
            layer_types=layer_types,
            attention_dropout=attention_dropout,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
        if isinstance(vision_encoder_config, dict):
            self.vision_encoder_config = self.sub_configs["vision_encoder_config"](**vision_encoder_config)
        elif isinstance(vision_encoder_config, PreTrainedConfig):
            self.vision_encoder_config = vision_encoder_config
        elif vision_encoder_config is None:
            self.vision_encoder_config = self.sub_configs["vision_encoder_config"]()
        else:
            raise ValueError(
                f"vision_encoder_config must be dict or PreTrainedConfig, got {type(vision_encoder_config)}."
            )

        self.image_token_id = image_token_id
        self.vision_projector_type = vision_projector_type
        self.tie_word_embeddings = tie_word_embeddings


# ===================== Vision Encoder =====================


class PenguinVLRMSNorm(Qwen3RMSNorm):
    pass


class PenguinVLMLP(Qwen3MLP):
    pass


class PenguinVLVisionEmbeddings(nn.Module):
    def __init__(self, config: PenguinVLVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.patch_size = config.patch_size
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.view(-1, self.config.num_channels, self.patch_size, self.patch_size)
        patch_embeds = self.patch_embedding(hidden_states)
        embeddings = patch_embeds.view(-1, self.embed_dim)
        return embeddings


class PenguinVLVisionRotaryEmbedding(nn.Module):
    """2D rotary position embedding for the vision encoder.

    Produces per-token ``(cos, sin)`` of shape ``(total_seq, head_dim)`` where
    the first ``head_dim / 2`` dimensions encode height positions and the last
    ``head_dim / 2`` dimensions encode width positions.  Uses ``rotate_half``
    coupling so that pair ``(i, i + head_dim/2)`` receives height rotation for
    ``i < head_dim/2`` and width rotation otherwise.
    """

    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: PenguinVLVisionConfig, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config

        self.rope_type = self.config.rope_parameters["rope_type"]
        rope_init_fn: Callable = self.compute_default_rope_parameters
        if self.rope_type != "default":
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = rope_init_fn(self.config, device)

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)

    @staticmethod
    def compute_default_rope_parameters(
        config: PenguinVLVisionConfig | None = None,
        device: Optional["torch.device"] = None,
        seq_len: int | None = None,
    ) -> tuple["torch.Tensor", float]:
        """
        Computes the inverse frequencies according to the original RoPE implementation
        Args:
            config ([`~transformers.PreTrainedConfig`]):
                The model configuration.
            device (`torch.device`):
                The device to use for initialization of the inverse frequencies.
            seq_len (`int`, *optional*):
                The current sequence length. Unused for this type of RoPE.
        Returns:
            Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
            post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
        """
        base = config.rope_parameters["rope_theta"]
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads

        attention_factor = 1.0  # Unused in this type of RoPE

        # Compute the inverse frequencies
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, attention_factor

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(2, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()  # shape (2, bs, 1, positions)

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def apply_multimodal_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    rope_section = [cos.shape[-1] // 2, cos.shape[-1] // 2]
    cos = torch.cat([m[i % 2] for i, m in enumerate(cos.split(rope_section, dim=-1))], dim=-1).unsqueeze(unsqueeze_dim)
    sin = torch.cat([m[i % 2] for i, m in enumerate(sin.split(rope_section, dim=-1))], dim=-1).unsqueeze(unsqueeze_dim)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class PenguinVLVisionAttention(nn.Module):
    """Multi-headed attention with QK normalization for the vision encoder."""

    def __init__(self, config: PenguinVLVisionConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = False

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = PenguinVLRMSNorm(self.head_dim, eps=config.rms_norm_eps)  # unlike olmo, only on the head dim!
        self.k_norm = PenguinVLRMSNorm(
            self.head_dim, eps=config.rms_norm_eps
        )  # thus post q_norm does not need reshape

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(query_states, key_states, cos, sin)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        if is_flash_attention_requested(self.config):
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask=None,
                scaling=self.scaling,
                dropout=0.0 if not self.training else self.attention_dropout,
                cu_seq_lens_q=cu_seqlens,
                cu_seq_lens_k=cu_seqlens,
                max_length_q=max_seqlen,
                max_length_k=max_seqlen,
                is_causal=False,
                **kwargs,
            )
        else:
            lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            splits = [
                torch.split(tensor, lengths.tolist(), dim=2) for tensor in (query_states, key_states, value_states)
            ]
            attn_outputs, attn_weights = [], []
            for q, k, v in zip(*splits):
                attn_output, attn_weight = attention_interface(
                    self,
                    q,
                    k,
                    v,
                    attention_mask=None,
                    scaling=self.scaling,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    is_causal=False,
                    **kwargs,
                )
                attn_outputs.append(attn_output)
                attn_weights.append(attn_weight)
            attn_output = torch.cat(attn_outputs, dim=1)

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class PenguinVLVisionEncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: PenguinVLVisionConfig, layer_idx: int):
        super().__init__()
        self.self_attn = PenguinVLVisionAttention(config, layer_idx)
        self.mlp = PenguinVLMLP(config)
        self.input_layernorm = PenguinVLRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = PenguinVLRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states,
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class PenguinVLVisionEncoder(nn.Module):
    def __init__(self, config: PenguinVLVisionConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [PenguinVLVisionEncoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = PenguinVLRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = PenguinVLVisionRotaryEmbedding(config=config)

    def get_rope_index(self, grid_sizes, merge_sizes, position_ids):
        position_ids = position_ids.contiguous()
        batch_size = grid_sizes.shape[0]

        # Vision Part: Generate 2D position indices for vision tokens
        vision_pos_ids = []
        for (t, h, w), merge_size in zip(grid_sizes, merge_sizes):
            # Generate height position indices
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w).to(position_ids.device)
            hpos_ids = hpos_ids.reshape(
                h // merge_size,
                merge_size,
                w // merge_size,
                merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            # Generate width position indices
            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1).to(position_ids.device)
            wpos_ids = wpos_ids.reshape(
                h // merge_size,
                merge_size,
                w // merge_size,
                merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()

            # Stack height and width to create 2D positions
            vision_pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))

        num_start_idx = 0
        for batch_idx in range(batch_size):
            pos_len = vision_pos_ids[batch_idx].shape[0]
            position_ids[:, 0, num_start_idx : num_start_idx + pos_len] = vision_pos_ids[batch_idx].permute(1, 0)
            num_start_idx += pos_len

        return position_ids

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        grid_thw: torch.Tensor,
        merge_sizes: torch.Tensor,
        **kwargs,
    ) -> tuple | BaseModelOutput:
        cache_position = torch.arange(0, hidden_states.shape[1], device=hidden_states.device)
        position_ids = cache_position.view(1, 1, -1).expand(2, hidden_states.shape[0], -1)
        position_ids = self.get_rope_index(grid_thw, merge_sizes, position_ids)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for encoder_layer in self.layers:
            hidden_states = encoder_layer(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutput(last_hidden_state=hidden_states)


class PenguinVLPreTrainedModel(PreTrainedModel):
    config_class = PenguinVLConfig
    supports_gradient_checkpointing = True
    _no_split_modules = ["PenguinVLVisionEncoderLayer"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True


class PenguinVLVisionModel(PenguinVLPreTrainedModel):
    config_class = PenguinVLVisionConfig
    main_input_name = "pixel_values"
    _can_record_outputs = {
        "hidden_states": PenguinVLVisionEncoderLayer,
        "attentions": PenguinVLVisionAttention,
    }

    def __init__(self, config: PenguinVLVisionConfig):
        super().__init__(config)
        self.embeddings = PenguinVLVisionEmbeddings(config)
        self.encoder = PenguinVLVisionEncoder(config)
        self.post_init()

    def get_input_embeddings(self) -> PenguinVLVisionEmbeddings:
        return self.embeddings.patch_embedding

    def pixel_unshuffle(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
        merge_sizes: torch.Tensor,
    ):
        hidden_states_chunks = hidden_states.split(grid_thw.prod(dim=1).tolist(), dim=0)
        outputs = []

        for hidden_states, (t, h, w), merge_size in zip(hidden_states_chunks, grid_thw, merge_sizes):
            c = hidden_states.shape[-1]
            hidden_states = hidden_states.view(t, h // merge_size, w // merge_size, merge_size, merge_size, c).permute(
                0, 1, 3, 2, 4, 5
            )
            hidden_states = hidden_states.reshape(t, h, w, c).permute(0, 3, 1, 2)
            hidden_states = F.interpolate(hidden_states, size=(h // merge_size, w // merge_size), mode="bilinear")
            hidden_states = hidden_states.permute(0, 2, 3, 1).view(-1, c)
            outputs.append(hidden_states)

        return torch.cat(outputs, dim=0)

    @capture_outputs(tie_last_hidden_states=False)
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        merge_sizes: torch.Tensor,
        **kwargs,
    ) -> tuple | BaseModelOutput:
        r"""
        grid_thw (`torch.LongTensor` of shape `(num_images_or_videos, 3)`):
            Temporal, height and width dimensions of the feature grid for each image/video.
        merge_sizes (`torch.Tensor` of shape `(num_images_or_videos,)`):
            Spatial downsampling ratio for each image or video.
        """
        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        hidden_states = self.embeddings(pixel_values.type(self.dtype))
        encoder_outputs: BaseModelOutput = self.encoder(
            hidden_states[None, ...],
            cu_seqlens=cu_seqlens,
            grid_thw=grid_thw,
            merge_sizes=merge_sizes,
            **kwargs,
        )

        last_hidden_state = encoder_outputs[0].squeeze(0)
        last_hidden_state = self.pixel_unshuffle(last_hidden_state, grid_thw, merge_sizes)

        return BaseModelOutput(last_hidden_state=last_hidden_state)


# ===================== Projector =====================


class PenguinVLProjector(nn.Module):
    def __init__(self, config: PenguinVLConfig):
        super().__init__()
        in_hidden_size = config.vision_encoder_config.hidden_size
        out_hidden_size = config.hidden_size

        projector_type = config.vision_projector_type
        mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", projector_type)
        if mlp_gelu_match:
            mlp_depth = int(mlp_gelu_match.group(1))
        else:
            raise ValueError(f"Unknown projector type: {projector_type}")

        modules = [nn.Linear(in_hidden_size, out_hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(out_hidden_size, out_hidden_size))
        self.readout = nn.Sequential(*modules)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.readout(hidden_states)


# ===================== Main Model =====================


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for PenguinVL outputs, with hidden states and attentions.
    """
)
class PenguinVLModelOutputWithPast(ModelOutput):
    r"""
    past_key_values (`Cache`, *optional*):
        Pre-computed hidden-states that can be used to speed up sequential decoding.
    image_hidden_states (`torch.FloatTensor`, *optional*):
        Hidden states produced by the vision encoder after projection.
    """

    last_hidden_state: torch.FloatTensor = None
    past_key_values: list[torch.FloatTensor] | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None
    attentions: tuple[torch.FloatTensor] | None = None
    image_hidden_states: torch.FloatTensor | None = None

class PenguinVLLanguageModel(Qwen3Model):
    pass


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for PenguinVL causal language model outputs.
    """
)
class PenguinVLCausalLMOutputWithPast(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
        Language modeling loss (for next-token prediction).
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
        Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    past_key_values (`Cache`, *optional*):
        Pre-computed hidden-states that can be used to speed up sequential decoding.
    image_hidden_states (`torch.FloatTensor`, *optional*):
        Hidden states produced by the vision encoder after projection.
    """

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    past_key_values: list[torch.FloatTensor] | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None
    attentions: tuple[torch.FloatTensor] | None = None
    image_hidden_states: torch.FloatTensor | None = None


class PenguinVLModel(PenguinVLPreTrainedModel):
    _checkpoint_conversion_mapping = {
        r"^vision_encoder\.vision_encoder\.": "vision_model.",
        r"^vision_encoder\.": "vision_model.",
        r"^vision_projector\.": "projector.",
        r"^embed_tokens\.": "language_model.embed_tokens.",
        r"^layers\.": "language_model.layers.",
        r"^norm\.": "language_model.norm.",
    }

    def __init__(self, config: PenguinVLConfig):
        super().__init__(config)
        self.vision_model = PenguinVLVisionModel._from_config(config.vision_encoder_config)
        self.projector = PenguinVLProjector(config)
        self.language_model = PenguinVLLanguageModel._from_config(config)
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    @can_return_tuple
    @auto_docstring(
        custom_intro="Obtains image last hidden states from the vision model and applies multimodal projection."
    )
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor,
        image_merge_sizes: torch.LongTensor,
        **kwargs,
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        pixel_values (`torch.FloatTensor`):
            Pixel values for the vision encoder.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`):
            Temporal, height and width of feature shape for each image.
        image_merge_sizes (`torch.Tensor` of shape `(num_images,)`):
            Spatial downsampling ratio for each image.
        """
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            grid_thw=image_grid_thw,
            merge_sizes=image_merge_sizes,
            return_dict=True,
            **kwargs,
        )
        last_hidden_state = vision_outputs.last_hidden_state
        image_embeds = self.projector(last_hidden_state)

        split_sizes = image_grid_thw.prod(dim=1) // (image_merge_sizes**2)
        image_embeds = torch.split(image_embeds, split_sizes.tolist())
        vision_outputs.pooler_output = image_embeds

        return vision_outputs

    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: torch.FloatTensor,
    ):
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = special_image_mask.all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id

        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        return special_image_mask

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        image_merge_sizes: torch.LongTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple | PenguinVLModelOutputWithPast:
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            Temporal, height and width of feature shape for each image.
        image_merge_sizes (`torch.Tensor` of shape `(num_images,)`, *optional*):
            Spatial downsampling ratio for each image.
        """
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_embeds = None
        if pixel_values is not None:
            image_embeds = self.get_image_features(
                pixel_values, image_grid_thw, image_merge_sizes, return_dict=True
            ).pooler_output
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds)
            num_mask_tokens = image_mask.sum() // inputs_embeds.shape[-1]
            num_image_embeds = image_embeds.shape[0]
            if num_mask_tokens != num_image_embeds:
                raise ValueError(
                    f"Number of image token positions ({num_mask_tokens}) does not match "
                    f"number of image embeddings ({num_image_embeds}). "
                    "Make sure the number of <image> tokens in your input matches the number of images/clips provided."
                )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        return PenguinVLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_embeds,
        )


class PenguinVLForConditionalGeneration(PenguinVLPreTrainedModel, GenerationMixin):
    _checkpoint_conversion_mapping = {
        r"^model\.vision_encoder\.vision_encoder\.": "model.vision_model.",
        r"^model\.vision_encoder\.": "model.vision_model.",
        r"^model\.vision_projector\.": "model.projector.",
        r"^model\.embed_tokens\.": "model.language_model.embed_tokens.",
        r"^model\.layers\.": "model.language_model.layers.",
        r"^model\.norm\.": "model.language_model.norm.",
    }
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}

    def __init__(self, config: PenguinVLConfig):
        super().__init__(config)
        self.model = PenguinVLModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    @can_return_tuple
    @auto_docstring
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor,
        image_merge_sizes: torch.LongTensor,
        **kwargs,
    ) -> tuple | BaseModelOutputWithPooling:
        return self.model.get_image_features(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            image_merge_sizes=image_merge_sizes,
            **kwargs,
        )

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        image_merge_sizes: torch.LongTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple | PenguinVLCausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[0, ..., config.vocab_size]`
            or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored (masked), the loss is
            only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            Temporal, height and width of feature shape for each image.
        image_merge_sizes (`torch.Tensor` of shape `(num_images,)`, *optional*):
            Spatial downsampling ratio for each image.
        """
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            image_merge_sizes=image_merge_sizes,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return PenguinVLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        image_merge_sizes: torch.LongTensor | None = None,
        is_first_iteration: bool | None = False,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            image_merge_sizes=image_merge_sizes,
            use_cache=use_cache,
            is_first_iteration=is_first_iteration,
            **kwargs,
        )

        if not is_first_iteration and use_cache:
            model_inputs["pixel_values"] = None

        return model_inputs


# ===================== Image Processor =====================


def _make_batched_clips(images) -> list[list]:
    """
    Normalize visual inputs to a list of clips, where each clip is a list of frames.

    - Single image: ``image`` -> ``[[image]]``
    - List of images: ``[img1, img2]`` -> ``[[img1], [img2]]``
    - Nested clips: ``[[img1], [f1, f2, f3]]`` -> ``[[img1], [f1, f2, f3]]``
    """
    if isinstance(images, list | tuple) and len(images) > 0:
        if isinstance(images[0], list | tuple):
            return [list(clip) for clip in images]
        if all(is_valid_image(f) for f in images):
            return [[img] for img in images]
    if is_valid_image(images):
        return [[images]]
    raise ValueError(f"Could not make batched images from {type(images)}")


def _simple_batched_resize(
    images,
    factor: int = 28,
    min_tokens: int = 16,
    max_tokens: int = 16384,
    input_data_format=None,
    frame_types=None,
):
    """
    Compute per-frame target ``(h, w)`` for a clip using TRA (Temporal Redundancy-Aware)
    token compression.

    Key frames (type 0) retain higher resolution. Intermediate frames (type 1) are
    allocated 1/16 of a key frame's area to reduce tokens while preserving temporal
    coverage. When all frames fit within the token budget, the original (aligned)
    resolution is kept for every frame.
    """
    min_pixels = min_tokens * factor * factor * 1.5
    max_pixels = max_tokens * factor * factor * 0.95

    first_image = images[0]
    if is_vision_available() and isinstance(first_image, Image.Image):
        width, height = first_image.size
    else:
        idf = input_data_format
        if idf is None:
            idf = infer_channel_dimension_format(first_image)
        height, width = get_image_size(first_image, channel_dim=idf)

    aspect_ratio = height / width
    raw_area = height * width
    num_frames = len(images)

    if frame_types is not None:
        ft_list = frame_types.tolist() if hasattr(frame_types, "tolist") else list(frame_types)
        num_key = ft_list.count(0)
        num_intermediate = ft_list.count(1)
    else:
        num_key = num_frames
        num_intermediate = 0
        ft_list = [0] * num_frames

    def _dims_from_area(target_area, ar, fac):
        w_new = math.sqrt(target_area / ar)
        h_new = w_new * ar
        return max(round(h_new / fac) * fac, fac), max(round(w_new / fac) * fac, fac)

    def _ensure_min(h, w, min_p, ar):
        if h * w < min_p:
            w_f = math.sqrt(min_p / ar)
            h_f = w_f * ar
            h = math.ceil(h_f / factor) * factor
            w = math.ceil(w_f / factor) * factor
        return h, w

    total_raw = num_frames * raw_area
    key_area = raw_area
    inter_area = raw_area

    if total_raw > max_pixels:
        eff = num_key + num_intermediate / 16.0
        key_area = max_pixels / eff
        inter_area = key_area / 16.0
        if inter_area < min_pixels:
            inter_area = min_pixels
            key_area = (max_pixels - num_intermediate * min_pixels) / max(num_key, 1)
        if key_area < min_pixels:
            key_area = min_pixels

    k_h, k_w = _dims_from_area(key_area, aspect_ratio, factor)
    k_h, k_w = _ensure_min(k_h, k_w, min_pixels, aspect_ratio)

    if num_intermediate > 0:
        i_h, i_w = _dims_from_area(inter_area, aspect_ratio, factor)
        i_h, i_w = _ensure_min(i_h, i_w, min_pixels, aspect_ratio)
    else:
        i_h, i_w = k_h, k_w

    return [(i_h, i_w) if ft_list[i] == 1 else (k_h, k_w) for i in range(num_frames)]


def _allocate_token_budget(clips, clip_merge_sizes, min_tokens, max_tokens, patch_size, input_data_format=None):
    """Distribute ``max_tokens`` across clips proportionally to their raw token counts."""
    clip_raw_tokens = []
    for clip, ms in zip(clips, clip_merge_sizes):
        first_frame = clip[0]
        if is_vision_available() and isinstance(first_frame, Image.Image):
            w, h = first_frame.size
        else:
            idf = input_data_format or infer_channel_dimension_format(first_frame)
            h, w = get_image_size(first_frame, channel_dim=idf)
        factor = patch_size * ms
        clip_raw_tokens.append(len(clip) * h * w / (factor * factor))

    total_raw = sum(clip_raw_tokens)
    if total_raw <= max_tokens:
        return [max_tokens] * len(clips)

    return [max(min_tokens * len(clip), raw * max_tokens / total_raw) for clip, raw in zip(clips, clip_raw_tokens)]


# ===================== KI Frame Extraction =====================

_KI_PATCH = 14
_KI_MIN_PIXELS = 10 * 14 * 14
_KI_MAX_PIXELS = 10240 * 14 * 14
_MIN_FRAME_SIMILARITY = 0.95


# Adapted from Keye-VL
def _get_frame_sim(
    frame1: torch.Tensor,
    frame2: torch.Tensor,
    patch_size: int = 14,
    threshold: float = 0.7,
    epsilon: float = 1e-8,
) -> float:
    """Cosine similarity between two frames averaged over patches. Returns mean similarity in [0, 1]."""

    def _to_comparison_tensor(tensor: torch.Tensor) -> torch.Tensor:
        if is_cv2_available():
            import cv2

            arr = tensor.cpu().permute(1, 2, 0).numpy()
            if arr.dtype in (np.float32, np.float64):
                arr = arr.astype(np.uint8)
            hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
            return torch.from_numpy(hsv).permute(2, 0, 1).to(tensor.device).float()
        return tensor.float()

    f1 = _to_comparison_tensor(frame1)
    f2 = _to_comparison_tensor(frame2)

    c, H, W = f1.shape
    h_patches = H // patch_size
    w_patches = W // patch_size

    def _to_patches(f):
        f = f[:, : h_patches * patch_size, : w_patches * patch_size]
        f = f.reshape(c, h_patches, patch_size, w_patches, patch_size)
        f = f.permute(1, 3, 0, 2, 4).reshape(h_patches, w_patches, c * patch_size * patch_size)
        return f.float()

    patch1 = _to_patches(f1)
    patch2 = _to_patches(f2)

    norm1 = torch.norm(patch1, p=2, dim=-1, keepdim=True) + epsilon
    norm2 = torch.norm(patch2, p=2, dim=-1, keepdim=True) + epsilon
    cos_sim = (patch1 / norm1 * patch2 / norm2).sum(dim=-1)

    both_near_zero = (norm1.squeeze(-1) < 0.01) & (norm2.squeeze(-1) < 0.01)
    similar = torch.ones_like(cos_sim)
    similar[~both_near_zero] = (cos_sim[~both_near_zero] > threshold).float()
    return similar[~both_near_zero].float().mean().item()


def _extract_ki_frames(
    frames: torch.Tensor,
    threshold: float = _MIN_FRAME_SIMILARITY,
) -> list:
    """
    Label each frame as keyframe (0) or non-keyframe (1) by comparing to the
    previous keyframe. First frame is always a keyframe; a new keyframe is chosen
    when similarity drops below threshold.
    """
    if frames.dim() != 4:
        raise ValueError("Frames must be 4D tensor [N, C, H, W]")
    if frames.size(0) <= 1:
        return [0] * frames.size(0)

    _, _, h, w = frames.shape
    rh, rw = smart_resize(h, w, factor=_KI_PATCH, min_pixels=_KI_MIN_PIXELS, max_pixels=_KI_MAX_PIXELS)
    resized = F.interpolate(frames, (rh, rw), mode="bilinear", antialias=True).float()

    indices = [0]
    key = resized[0]
    for i in range(1, resized.size(0)):
        if _get_frame_sim(key, resized[i]) < threshold:
            indices.append(i)
            key = resized[i]

    frame_types = torch.ones(frames.size(0), dtype=torch.int32)
    frame_types[indices] = 0
    return frame_types.tolist()


class PenguinVLImageProcessorKwargs(Qwen2VLImageProcessorKwargs, total=False):
    merge_size: int | list[int]
    frame_types: list | None


class PenguinVLImageProcessor(Qwen2VLImageProcessor):
    r"""
    Image processor for PenguinVL with dynamic resizing and TRA (Temporal Redundancy-Aware)
    token compression for video frames.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BICUBIC`):
            Resampling filter to use when resizing.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by `rescale_factor`.
        rescale_factor (`float`, *optional*, defaults to `1/255`):
            Scale factor for rescaling.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image.
        image_mean (`list[float]`, *optional*, defaults to `[0.5, 0.5, 0.5]`):
            Mean for normalization.
        image_std (`list[float]`, *optional*, defaults to `[0.5, 0.5, 0.5]`):
            Standard deviation for normalization.
        do_convert_rgb (`bool`, *optional*, defaults to `True`):
            Whether to convert the image to RGB.
        min_pixels (`int`, *optional*, defaults to 3136):
            Minimum pixels for resizing (equivalent to ``min_tokens * patch_size ** 2``).
        max_pixels (`int`, *optional*, defaults to 3211264):
            Maximum pixels for resizing (equivalent to ``max_tokens * patch_size ** 2``).
        patch_size (`int`, *optional*, defaults to 14):
            Spatial patch size of the vision encoder.
        merge_size (`int`, *optional*, defaults to 1):
            Default spatial merge size for token compression (1 for images, 2 for video).
    """

    model_input_names = ["pixel_values", "image_grid_thw", "image_merge_sizes"]
    valid_kwargs = PenguinVLImageProcessorKwargs

    def __init__(
        self,
        do_resize: bool = True,
        size: dict[str, int] | None = None,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: int | float = 1 / 255,
        do_normalize: bool = True,
        image_mean: float | list[float] | None = None,
        image_std: float | list[float] | None = None,
        do_convert_rgb: bool = True,
        min_pixels: int = 3136,
        max_pixels: int = 3211264,
        patch_size: int = 14,
        temporal_patch_size: int = 1,
        merge_size: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(
            do_resize=do_resize,
            size=size,
            resample=resample,
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_convert_rgb=do_convert_rgb,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            merge_size=merge_size,
            **kwargs,
        )

        self.image_mean = image_mean if image_mean is not None else IMAGENET_STANDARD_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_STANDARD_STD

        if self.temporal_patch_size != 1:
            raise ValueError("`temporal_patch_size` must be 1 for PenguinVL")

    def preprocess(
        self,
        images: ImageInput,
        do_resize: bool | None = None,
        size: dict[str, int] | None = None,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        resample: PILImageResampling = None,
        do_rescale: bool | None = None,
        rescale_factor: float | None = None,
        do_normalize: bool | None = None,
        image_mean: float | list[float] | None = None,
        image_std: float | list[float] | None = None,
        do_convert_rgb: bool | None = None,
        merge_size: int | list[int] | None = None,
        frame_types: list | None = None,
        return_tensors: str | TensorType | None = None,
        data_format: ChannelDimension | None = ChannelDimension.FIRST,
        input_data_format: str | ChannelDimension | None = None,
    ):
        """
        Preprocess images or video clips with optional TRA key/intermediate frame compression.

        Args:
            images: Single image, list of images, or nested ``[[clip1_frames], [clip2_frames]]``.
            merge_size: Spatial merge size. Can be ``int`` (all clips) or ``list[int]`` (per-clip).
                Typically 1 for images and 2 for video.
            frame_types: Per-clip frame type annotations. ``None`` means all key frames.
                Each clip's frame_types is a list where 0 = key frame, 1 = intermediate frame.
                Pass as ``[ft_clip1, ft_clip2, ...]`` or ``[ft_single_clip]``.
        """
        min_pixels = min_pixels if min_pixels is not None else self.min_pixels
        max_pixels = max_pixels if max_pixels is not None else self.max_pixels

        if size is not None:
            if "shortest_edge" not in size or "longest_edge" not in size:
                raise ValueError("size must contain 'shortest_edge' and 'longest_edge' keys.")
            min_pixels = size["shortest_edge"]
        elif min_pixels is not None and max_pixels is not None:
            # backward compatibility: override size with min_pixels and max_pixels if they are provided
            size = {"shortest_edge": min_pixels, "longest_edge": max_pixels}
        else:
            size = {**self.size}
        do_resize = do_resize if do_resize is not None else self.do_resize
        resample = resample if resample is not None else self.resample
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        default_merge = merge_size if merge_size is not None else self.merge_size
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb

        validate_preprocess_arguments(
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_resize=do_resize,
            size=size,
            resample=resample,
        )

        clips = _make_batched_clips(images)
        num_clips = len(clips)

        if isinstance(default_merge, list | tuple):
            clip_merge_sizes = list(default_merge)
        else:
            clip_merge_sizes = [default_merge] * num_clips

        if frame_types is None:
            clip_frame_types = [None] * num_clips
        elif isinstance(frame_types, list | tuple) and len(frame_types) > 0:
            if isinstance(frame_types[0], list | tuple) or frame_types[0] is None:
                clip_frame_types = list(frame_types)
            else:
                clip_frame_types = [frame_types] if num_clips == 1 else [None] * num_clips
        else:
            clip_frame_types = [None] * num_clips

        ps2 = self.patch_size * self.patch_size
        clip_budgets = _allocate_token_budget(
            clips,
            clip_merge_sizes,
            min_tokens=self.min_pixels // ps2,
            max_tokens=self.max_pixels // ps2,
            patch_size=self.patch_size,
            input_data_format=input_data_format,
        )

        pixel_values_list = []
        grid_thw_list = []
        merge_sizes_list = []
        num_frames_per_clip = []

        for clip, ms, ft, budget in zip(clips, clip_merge_sizes, clip_frame_types, clip_budgets):
            factor = self.patch_size * ms
            target_sizes = _simple_batched_resize(
                clip,
                factor=factor,
                min_tokens=self.min_pixels // ps2,
                max_tokens=budget,
                input_data_format=input_data_format,
                frame_types=ft,
            )

            clip_n = 0
            for frame, target_size in zip(clip, target_sizes):
                frame_convert_rgb = do_convert_rgb
                frame_data_fmt = input_data_format
                if do_resize:
                    if do_convert_rgb:
                        frame = convert_to_rgb(frame)
                    frame = to_numpy_array(frame)
                    if frame_data_fmt is None:
                        frame_data_fmt = infer_channel_dimension_format(frame)
                    rh, rw = int(target_size[0]), int(target_size[1])
                    frame = resize(frame, size=(rh, rw), resample=resample, input_data_format=frame_data_fmt)
                    frame_convert_rgb = False

                patches, grid_thw = self._preprocess(
                    frame,
                    do_resize=False,
                    size=size,
                    resample=resample,
                    do_rescale=do_rescale,
                    rescale_factor=rescale_factor,
                    do_normalize=do_normalize,
                    image_mean=image_mean,
                    image_std=image_std,
                    patch_size=self.patch_size,
                    temporal_patch_size=1,
                    merge_size=ms,
                    do_convert_rgb=frame_convert_rgb,
                    data_format=data_format,
                    input_data_format=frame_data_fmt,
                )
                pixel_values_list.append(patches)
                grid_thw_list.append(grid_thw)
                merge_sizes_list.append(ms)
                clip_n += 1
            num_frames_per_clip.append(clip_n)

        pixel_values = np.concatenate(pixel_values_list, axis=0)
        image_grid_thw = np.array(grid_thw_list)
        image_merge_sizes = np.array(merge_sizes_list)

        data = {
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "image_merge_sizes": image_merge_sizes,
            "num_frames_per_clip": num_frames_per_clip,
        }
        return BatchFeature(data=data, tensor_type=return_tensors)


class PenguinVLImageProcessorFast(Qwen2VLImageProcessorFast):
    r"""
    Fast image processor for PenguinVL with dynamic per-clip resizing and TRA (Temporal
    Redundancy-Aware) token compression for video frames.

    Compared to the base Qwen2-VL fast processor this class:

    * Supports **per-clip merge sizes** (``merge_size`` may be ``int`` or ``list[int]``).
    * Applies TRA compression: key frames retain high resolution while intermediate
      frames are allocated ~1/16 of the tokens.
    * Returns ``image_merge_sizes`` and ``num_frames_per_clip`` alongside pixel values.
    """

    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    temporal_patch_size = 1
    valid_kwargs = PenguinVLImageProcessorKwargs
    model_input_names = ["pixel_values", "image_grid_thw", "image_merge_sizes"]

    def _preprocess_image_like_inputs(
        self,
        images: ImageInput,
        do_convert_rgb: bool,
        input_data_format: ChannelDimension,
        device: Union[str, "torch.device"] | None = None,
        **kwargs,
    ) -> BatchFeature:
        if kwargs["temporal_patch_size"] != 1:
            raise ValueError("`temporal_patch_size` must be 1 for PenguinVL")

        merge_size_param = kwargs.pop("merge_size")
        frame_types_param = kwargs.pop("frame_types", None)
        size = kwargs["size"]
        patch_size = kwargs["patch_size"]
        do_resize = kwargs["do_resize"]
        interpolation = kwargs["interpolation"]
        do_rescale = kwargs["do_rescale"]
        rescale_factor = kwargs["rescale_factor"]
        do_normalize = kwargs["do_normalize"]
        image_mean = kwargs["image_mean"]
        image_std = kwargs["image_std"]
        return_tensors = kwargs.get("return_tensors")

        min_pixels = size["shortest_edge"]
        max_pixels = size["longest_edge"]

        clips = _make_batched_clips(images)
        num_clips = len(clips)

        if isinstance(merge_size_param, (list, tuple)):
            clip_merge_sizes = list(merge_size_param)
        else:
            clip_merge_sizes = [merge_size_param] * num_clips

        if frame_types_param is None:
            clip_frame_types = [None] * num_clips
        elif isinstance(frame_types_param, (list, tuple)) and len(frame_types_param) > 0:
            if isinstance(frame_types_param[0], (list, tuple)) or frame_types_param[0] is None:
                clip_frame_types = list(frame_types_param)
            else:
                clip_frame_types = [frame_types_param] if num_clips == 1 else [None] * num_clips
        else:
            clip_frame_types = [None] * num_clips

        ps2 = patch_size * patch_size
        min_tokens = min_pixels // ps2
        max_tokens = max_pixels // ps2
        clip_budgets = _allocate_token_budget(
            clips,
            clip_merge_sizes,
            min_tokens,
            max_tokens,
            patch_size,
        )

        pixel_values_list = []
        grid_thw_list = []
        merge_sizes_list = []
        num_frames_per_clip = []

        for clip, ms, ft, budget in zip(clips, clip_merge_sizes, clip_frame_types, clip_budgets):
            factor = patch_size * ms
            target_sizes = _simple_batched_resize(
                clip,
                factor=factor,
                min_tokens=min_tokens,
                max_tokens=budget,
                input_data_format=input_data_format,
                frame_types=ft,
            )

            clip_n = 0
            for frame, target_size in zip(clip, target_sizes):
                frame_tensor = self._process_image(
                    frame,
                    do_convert_rgb=do_convert_rgb,
                    input_data_format=input_data_format,
                    device=device,
                )

                if do_resize:
                    frame_tensor = self.resize(
                        frame_tensor,
                        size=SizeDict(height=int(target_size[0]), width=int(target_size[1])),
                        interpolation=interpolation,
                    )

                frame_tensor = self.rescale_and_normalize(
                    frame_tensor.unsqueeze(0),
                    do_rescale,
                    rescale_factor,
                    do_normalize,
                    image_mean,
                    image_std,
                )

                resized_height, resized_width = frame_tensor.shape[-2:]
                grid_h = resized_height // patch_size
                grid_w = resized_width // patch_size
                channel = frame_tensor.shape[-3]

                patches = frame_tensor.view(
                    1,
                    1,
                    1,
                    channel,
                    grid_h // ms,
                    ms,
                    patch_size,
                    grid_w // ms,
                    ms,
                    patch_size,
                )
                patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
                flatten_patches = patches.reshape(
                    grid_h * grid_w,
                    channel * patch_size * patch_size,
                )

                pixel_values_list.append(flatten_patches)
                grid_thw_list.append([1, grid_h, grid_w])
                merge_sizes_list.append(ms)
                clip_n += 1

            num_frames_per_clip.append(clip_n)

        pixel_values = torch.cat(pixel_values_list, dim=0)
        image_grid_thw = torch.tensor(grid_thw_list)
        image_merge_sizes = torch.tensor(merge_sizes_list)

        return BatchFeature(
            data={
                "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thw,
                "image_merge_sizes": image_merge_sizes,
                "num_frames_per_clip": num_frames_per_clip,
            },
            tensor_type=return_tensors,
        )


# ===================== Processor =====================


class PenguinVLProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
    }


class PenguinVLProcessor(ProcessorMixin):
    r"""
    Processor for PenguinVL that wraps an image processor and a tokenizer.

    Args:
        image_processor (`PenguinVLImageProcessor`):
            The image processor.
        tokenizer (`PreTrainedTokenizer`):
            The tokenizer.
        image_token (`str`, *optional*, defaults to `" "`):
            The image placeholder token.
        image_merge_size (`int`, *optional*, defaults to 1):
            Spatial merge size for images.
        video_merge_size (`int`, *optional*, defaults to 2):
            Spatial merge size for video frames.
        chat_template (`str`, *optional*):
            A Jinja template for formatting conversations.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "PenguinVLImageProcessor"
    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")
    valid_kwargs = ["chat_template", "image_token", "image_merge_size", "video_merge_size"]

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        image_token="<image>",
        image_merge_size: int = 1,
        video_merge_size: int = 2,
        chat_template=None,
        **kwargs,
    ):
        self.image_token = image_token
        self.image_merge_size = image_merge_size
        self.video_merge_size = video_merge_size
        if tokenizer is not None:
            self.image_token_id = tokenizer.convert_tokens_to_ids(image_token)
        super().__init__(image_processor=image_processor, tokenizer=tokenizer, chat_template=chat_template, **kwargs)

    def __call__(
        self,
        images: ImageInput = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] = None,
        frame_types: list | None = None,
        **kwargs: Unpack[PenguinVLProcessorKwargs],
    ) -> BatchFeature:
        output_kwargs = self._merge_kwargs(
            PenguinVLProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        image_inputs = {}
        num_frames_per_clip = None
        if images is not None:
            # Load images from URLs if needed (e.g. from apply_chat_template with return_dict=True)
            def _load_if_url(x):
                if isinstance(x, str) and (x.startswith("http://") or x.startswith("https://")):
                    return load_image(x)
                return x

            def _load_images(imgs):
                if isinstance(imgs, (list, tuple)):
                    return [_load_images(item) for item in imgs]
                return _load_if_url(imgs)

            images = _load_images(images)
            clips = _make_batched_clips(images)
            merge_size = [self.video_merge_size if len(clip) > 1 else self.image_merge_size for clip in clips]
            images_kwargs = {**output_kwargs.get("images_kwargs", {}), "merge_size": merge_size}
            if frame_types is not None:
                images_kwargs["frame_types"] = frame_types
            image_inputs = self.image_processor(images=images, **images_kwargs)
            image_grid_thw = image_inputs["image_grid_thw"]
            image_merge_sizes = image_inputs["image_merge_sizes"]
            num_frames_per_clip = image_inputs.pop("num_frames_per_clip", None)
        else:
            image_grid_thw = image_merge_sizes = []

        if not isinstance(text, list):
            text = [text]

        text = text.copy()

        if images is not None:
            total_image_tokens_in_text = sum(t.count(self.image_token) for t in text)
            total_frames = int(sum(num_frames_per_clip)) if num_frames_per_clip is not None else len(image_grid_thw)

            if total_image_tokens_in_text == total_frames:
                frame_idx = 0
                for i in range(len(text)):
                    while self.image_token in text[i]:
                        t, h, w = image_grid_thw[frame_idx]
                        ms = image_merge_sizes[frame_idx]
                        num_image_tokens = int(t * (h // ms) * (w // ms))
                        text[i] = text[i].replace(self.image_token, "<|placeholder|>" * num_image_tokens, 1)
                        frame_idx += 1
                    text[i] = text[i].replace("<|placeholder|>", self.image_token)
            else:
                frame_idx = 0
                clip_idx = 0
                for i in range(len(text)):
                    while self.image_token in text[i]:
                        n_frames = num_frames_per_clip[clip_idx] if num_frames_per_clip is not None else 1
                        num_image_tokens = 0
                        for j in range(n_frames):
                            t, h, w = image_grid_thw[frame_idx + j]
                            ms = image_merge_sizes[frame_idx + j]
                            num_image_tokens += int(t * (h // ms) * (w // ms))
                        text[i] = text[i].replace(self.image_token, "<|placeholder|>" * num_image_tokens, 1)
                        frame_idx += n_frames
                        clip_idx += 1
                    text[i] = text[i].replace("<|placeholder|>", self.image_token)

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"], return_tensors=None)

        return BatchFeature(data={**text_inputs, **image_inputs}, tensor_type=return_tensors)

    def _load_visual(self, source):
        """Load a single image from URL, file:// path, local path, or pass through PIL images."""
        if isinstance(source, str):
            source = source.removeprefix("file://")
            return load_image(source)
        if is_vision_available() and isinstance(source, Image.Image):
            return source
        return source

    def _load_video_frames(self, video_url, fps=1, max_frames=128):
        """
        Load frames from a video with fps-based sampling capped at max_frames,
        then extract KI (key/intermediate) frame types.

        Sampling logic:
        - Read at ``fps`` frames per second (default 1).
        - If the resulting frame count exceeds ``max_frames``, uniformly
          subsample to ``max_frames`` frames.

        Returns:
            tuple: ``(frames, frame_types, timestamps)`` where *frames* is a
            list of PIL images, *frame_types* is a list of ints (0 = keyframe,
            1 = intermediate frame), and *timestamps* is a list of floats
            (seconds) for each sampled frame.
        """
        from ...video_utils import load_video

        _BACKEND_PRIORITY = ("decord", "opencv", "torchvision", "torchcodec", "pyav")
        _BACKEND_AVAILABLE = {
            "pyav": is_av_available,
            "decord": is_decord_available,
            "opencv": is_cv2_available,
            "torchvision": is_torchvision_available,
            "torchcodec": is_torchcodec_available,
        }
        backend = next(
            (b for b in _BACKEND_PRIORITY if _BACKEND_AVAILABLE[b]()),
            None,
        )
        if backend is None:
            raise ImportError(
                "No video backend available. Install one of: av (pyav), decord, opencv-python, torchvision, or torchcodec."
            )

        _fps = fps
        _max = max_frames
        _sampled_indices = []
        _video_fps = [30.0]

        def _sample_fn(metadata, **kwargs):
            total = metadata.total_num_frames
            video_fps = metadata.fps or 30.0
            _video_fps[0] = video_fps
            if total <= 0:
                # Frame count unknown (not stored in container header); take consecutive frames up to _max
                indices = np.arange(0, _max, dtype=int)
            else:
                num_at_target_fps = max(1, int(total / video_fps * _fps))
                if num_at_target_fps <= _max:
                    indices = np.arange(0, total, max(1, total / num_at_target_fps), dtype=int)
                else:
                    indices = np.linspace(0, total - 1, _max, dtype=int)
            indices = indices[:_max]
            _sampled_indices.extend(indices.tolist())
            return indices

        video_frames, _ = load_video(video_url, sample_indices_fn=_sample_fn, backend=backend)

        if hasattr(video_frames, "numpy"):
            video_frames = video_frames.numpy()
        if not isinstance(video_frames, np.ndarray):
            video_frames = np.stack([np.array(f) for f in video_frames])

        frames_tensor = torch.from_numpy(video_frames.transpose(0, 3, 1, 2).copy()).float()
        frame_types = _extract_ki_frames(frames_tensor)
        timestamps = [idx / _video_fps[0] for idx in _sampled_indices]

        if is_vision_available():
            frames = [Image.fromarray(video_frames[i]) for i in range(len(video_frames))]
        else:
            frames = list(video_frames)

        return frames, frame_types, timestamps

    def _convert_messages_for_chat_template(self, messages):
        """
        Convert Qwen2-VL style messages for the Jinja chat template.

        Image entries become ``{"type": "image"}``.  Video entries keep their
        type and carry ``num_frames`` / ``timestamps`` so the template can emit
        per-frame timestamp prefixes.  Call :meth:`process_vision_info` before
        :meth:`apply_chat_template` to populate these fields automatically.

        If ``num_frames`` is not present on a video entry (i.e.
        :meth:`process_vision_info` was not called first), the entry falls back
        to a plain ``{"type": "image"}`` for backward compatibility.
        """
        converted = copy.deepcopy(messages)
        for message in converted:
            content = message.get("content", [])
            if isinstance(content, str):
                continue
            new_content = []
            for item in content:
                if not isinstance(item, dict):
                    new_content.append(item)
                    continue
                if item.get("type") == "image":
                    new_content.append({"type": "image"})
                elif item.get("type") == "video":
                    if "num_frames" in item:
                        video_entry = {"type": "video", "num_frames": item["num_frames"]}
                        if "timestamps" in item:
                            video_entry["timestamps"] = item["timestamps"]
                        new_content.append(video_entry)
                    else:
                        new_content.append({"type": "image"})
                else:
                    new_content.append(item)
            message["content"] = new_content
        return converted

    def process_vision_info(
        self,
        messages: list[dict],
        fps: int = 1,
        max_frames: int = 128,
    ) -> tuple[list, list] | tuple[None, None]:
        """
        Extract and load visual inputs from Qwen2-VL style conversation messages.

        Walks through ``messages`` and collects images / video frames in order.
        For video clips, frames are sampled at ``fps`` (default 1) and capped at
        ``max_frames`` (default 128), then KI frame types are extracted.

        Video content items in ``messages`` are enriched in-place with
        ``num_frames`` and ``timestamps`` keys so that a subsequent call to
        :meth:`apply_chat_template` can emit per-frame timestamp prefixes.
        Call this method **before** :meth:`apply_chat_template`.

        Supported content block formats::

            {"type": "image", "image": "https://example.com/photo.jpg"}
            {"type": "image", "image": "file:///path/to/image.png"}
            {"type": "image", "image": <PIL.Image.Image>}
            {"type": "video", "video": "https://example.com/clip.mp4"}
            {"type": "video", "video": ["file:///path/frame1.jpg", ...], "timestamps": [0, ...]}
            {"type": "video", "video": [<PIL.Image.Image>, ...], "timestamps": [0, ...]}

        Args:
            messages: Conversation in Qwen2-VL dict format. Video content items
                are enriched in-place with ``num_frames`` and ``timestamps``.
            fps: Frames per second for video sampling. Defaults to 1.
            max_frames: Maximum number of frames per video. Defaults to 128.

        Returns:
            ``(visual_inputs, clip_frame_types)`` where *visual_inputs* is a
            nested list of PIL images and *clip_frame_types* is a list of
            per-clip frame type annotations (``None`` for images, ``list[int]``
            for videos where 0 = keyframe, 1 = intermediate frame). Returns
            ``(None, None)`` when no visual content is found.

        Example::

            images, frame_types = processor.process_vision_info(messages)
            text = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(images=images, text=text, frame_types=frame_types, return_tensors="pt")
        """
        visual_inputs = []
        clip_frame_types = []
        for message in messages:
            content = message.get("content", [])
            if isinstance(content, str):
                continue
            for item in content:
                if not isinstance(item, dict):
                    continue
                content_type = item.get("type")
                if content_type == "image":
                    source = item.get("image") or item.get("url") or item.get("path")
                    if source is not None:
                        img = self._load_visual(source)
                        visual_inputs.append([img])
                        clip_frame_types.append(None)
                elif content_type == "video":
                    video_data = item.get("video") or item.get("url") or item.get("path")
                    if video_data is None:
                        continue
                    if isinstance(video_data, (list, tuple)):
                        frames = [self._load_visual(f) for f in video_data]
                        np_frames = np.stack([np.array(f) for f in frames])
                        ft_tensor = torch.from_numpy(np_frames.transpose(0, 3, 1, 2).copy()).float()
                        ft = _extract_ki_frames(ft_tensor)
                        visual_inputs.append(frames)
                        clip_frame_types.append(ft)
                        item["num_frames"] = len(frames)
                        if "timestamps" not in item:
                            item["timestamps"] = []
                    elif isinstance(video_data, str):
                        frames, ft, timestamps = self._load_video_frames(video_data, fps=fps, max_frames=max_frames)
                        visual_inputs.append(frames)
                        clip_frame_types.append(ft)
                        item["num_frames"] = len(frames)
                        if "timestamps" not in item:
                            item["timestamps"] = timestamps

        if not visual_inputs:
            return None, None
        return visual_inputs, clip_frame_types

    def apply_chat_template(self, conversation, chat_template=None, **kwargs):
        kwargs.setdefault("image_token", self.image_token)
        conversation = self._convert_messages_for_chat_template(conversation)
        return super().apply_chat_template(conversation, chat_template, **kwargs)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))


__all__ = [
    "PenguinVLVisionConfig",
    "PenguinVLConfig",
    "PenguinVLVisionModel",
    "PenguinVLPreTrainedModel",
    "PenguinVLModel",
    "PenguinVLForConditionalGeneration",
    "PenguinVLProcessor",
    "PenguinVLImageProcessor",
    "PenguinVLImageProcessorFast",
]
