# Copyright 2026 Xiaomi Corporation and the HuggingFace Inc. team. All rights reserved.
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
"""PyTorch MiMo V2 VL model: image, video and audio understanding on top of the MiMo V2 hybrid MoE decoder."""

import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from itertools import groupby

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict

from ... import initialization as init
from ...audio_utils import AudioInput, make_list_of_audio, mel_filter_bank, spectrogram, window_function
from ...cache_utils import Cache
from ...configuration_utils import PreTrainedConfig, SubConfigSpec
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...generation import GenerationMixin
from ...image_processing_backends import TorchvisionBackend
from ...image_utils import ChannelDimension, ImageInput, PILImageResampling, SizeDict
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    BaseModelOutputWithPooling,
    CausalLMOutputWithPast,
)
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, slice_logits_to_keep
from ...processing_utils import AudioKwargs, ProcessingKwargs, ProcessorMixin, Unpack, VideosKwargs
from ...tokenization_utils_base import TextInput
from ...utils import (
    ModelOutput,
    TensorType,
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
    is_torch_available,
    is_torch_tensor,
    torch_compilable_check,
)
from ...utils.generic import accepts_precomputed_kwargs
from ...utils.output_capturing import capture_outputs
from ...video_processing_utils import BaseVideoProcessor
from ...video_utils import (
    VideoInput,
    VideoMetadata,
    group_videos_by_shape,
    make_batched_metadata,
    make_batched_videos,
    reorder_videos,
)
from ...vision_utils import get_vision_attention_seqlens, get_vision_position_ids
from ..encodec.modeling_encodec import EncodecEuclideanCodebook, EncodecVectorQuantization
from ..glmasr.configuration_glmasr import GlmAsrEncoderConfig
from ..glmasr.modeling_glmasr import GlmAsrAttention, GlmAsrEncoderLayer, GlmAsrMLP
from ..mimo_v2_flash.configuration_mimo_v2_flash import MiMoV2FlashConfig
from ..mimo_v2_flash.modeling_mimo_v2_flash import (
    MiMoV2FlashAttention,
    MiMoV2FlashDecoderLayer,
    MiMoV2FlashModel,
    MiMoV2FlashPreTrainedModel,
    MiMoV2FlashRMSNorm,
    MiMoV2FlashRotaryEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from ..olmo.modeling_olmo import OlmoRotaryEmbedding
from ..qwen2.modeling_qwen2 import Qwen2Attention, Qwen2DecoderLayer, Qwen2MLP, Qwen2RotaryEmbedding
from ..qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig, Qwen2_5_VLVisionConfig
from ..qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionPatchEmbed,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLMLP,
    Qwen2_5_VLModel,
    Qwen2_5_VLVisionBlock,
    Qwen2_5_VLVisionRotaryEmbedding,
    apply_rotary_pos_emb_vision,
)
from ..qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor, Qwen2VLImageProcessorKwargs
from ..qwen2_vl.video_processing_qwen2_vl import Qwen2VLVideoProcessor, Qwen2VLVideoProcessorInitKwargs
from ..qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeTalkerResizeMLP
from ..qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionPatchMerger


@auto_docstring(checkpoint="XiaomiMiMo/MiMo-V2.6-Flash-RL")
@strict
class MiMoV2VLTextConfig(MiMoV2FlashConfig):
    r"""
    n_group (`int`, *optional*, defaults to 1):
        Number of expert groups for group-based top-k routing.
    topk_group (`int`, *optional*, defaults to 1):
        Number of groups selected per token in group-based top-k routing.
    head_dim (`int`, *optional*, defaults to 192):
        Dimension of query and key heads.
    v_head_dim (`int`, *optional*, defaults to 128):
        Dimension of value heads.
    mlp_layer_types (`list`, *optional*):
        MLP pattern for each layer (`"dense"` or `"sparse"`). Defaults to 1 dense + rest sparse.
    attention_value_scale (`float`, *optional*, defaults to 0.707):
        Multiplier applied to value projections before attention. `None` disables scaling.
    swa_num_attention_heads (`int`, *optional*):
        Number of query heads in sliding attention. Defaults to `num_attention_heads`.
    swa_num_key_value_heads (`int`, *optional*):
        Number of sliding-attention KV heads. Defaults to twice `num_key_value_heads`.
    swa_head_dim (`int`, *optional*):
        Sliding-attention query/key head dimension. Defaults to `head_dim`.
    swa_v_head_dim (`int`, *optional*):
        Sliding-attention value head dimension. Defaults to `v_head_dim`.
    add_full_attention_sink_bias (`bool`, *optional*, defaults to `False`):
        Whether global attention layers have learned softmax sinks.
    add_swa_attention_sink_bias (`bool`, *optional*, defaults to `True`):
        Whether sliding attention layers have learned softmax sinks.
    """

    model_type = "mimo_v2_vl_text"
    base_config_key = "text_config"
    # The router logits are not returned: there is no auxiliary load-balancing loss.
    output_router_logits = AttributeError()

    max_position_embeddings: int = 1048576
    rms_norm_eps: float = 1e-6
    bos_token_id: int | None = None
    pad_token_id: int | None = 151643
    eos_token_id: int | list[int] | None = 151645

    def __post_init__(self, **kwargs):
        # Per-layer rope defaults matching the MiMo V2.6 pretrained thetas.
        if self.rope_parameters is None:
            self.rope_parameters = {
                "full_attention": {"rope_type": "default", "rope_theta": 10_000_000.0, "partial_rotary_factor": 0.334},
                "sliding_attention": {"rope_type": "default", "rope_theta": 10_000.0, "partial_rotary_factor": 0.334},
            }
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        for prefix in ("", "swa_"):
            num_heads = getattr(self, f"{prefix}num_attention_heads")
            num_key_value_heads = getattr(self, f"{prefix}num_key_value_heads")
            if num_heads % num_key_value_heads != 0:
                raise ValueError(
                    f"`{prefix}num_attention_heads` ({num_heads}) must be divisible by "
                    f"`{prefix}num_key_value_heads` ({num_key_value_heads})."
                )
        if self.n_routed_experts % self.n_group != 0:
            raise ValueError(f"`n_routed_experts` ({self.n_routed_experts}) must be divisible by `n_group`.")


@auto_docstring(checkpoint="XiaomiMiMo/MiMo-V2.6-Flash-RL")
@strict
class MiMoV2VLVisionConfig(Qwen2_5_VLVisionConfig):
    r"""
    num_key_value_heads (`int`, *optional*, defaults to 8):
        Number of key/value heads of the grouped-query attention.
    head_dim (`int`, *optional*, defaults to 64):
        Dimension of the attention heads, independent of `hidden_size // num_heads`.
    out_hidden_size (`int`, *optional*, defaults to 4096):
        Output dimension of the patch merger; must match the language model's hidden size.
    layer_types (`list[str]`, *optional*):
        Attention type of each block: `"full_attention"` (every patch of the image or frame) or `"sliding_attention"`
        (the `sliding_window` patches on each side, with learned attention sinks).
    sliding_window (`int`, *optional*, defaults to 64):
        Number of patches visible on each side of a query in `"sliding_attention"` blocks.
    patch_orders (`list[str]`, *optional*):
        Order in which each block traverses the patches of every image or frame: `"row_major"` (as produced by the
        processor) or `"column_major"`. This matters for the windows of `"sliding_attention"` blocks.
    """

    model_type = "mimo_v2_vl_vision"
    attribute_map = {"num_attention_heads": "num_heads", "num_hidden_layers": "depth"}

    depth: int = 28
    hidden_size: int = 1280
    intermediate_size: int = 4608
    num_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 64
    patch_size: int = 16
    temporal_patch_size: int = 2
    out_hidden_size: int = 4096
    rms_norm_eps: float = 1e-6
    layer_types: list[str] | None = None
    sliding_window: int = 64
    patch_orders: list[str] | None = None
    # Replaced by `layer_types`, `sliding_window` and `patch_orders`; the language model uses 1D positions.
    tokens_per_second = AttributeError()
    window_size = AttributeError()
    fullatt_block_indexes = AttributeError()

    def __post_init__(self, **kwargs):
        # Blocks 0, 9, 18 and 27 attend to the full image. Within each cycle of 9 blocks, the 4 last sliding blocks
        # traverse the patches column by column.
        if self.layer_types is None:
            self.layer_types = [
                "full_attention" if i % 9 == 0 else "sliding_attention" for i in range(self.num_hidden_layers)
            ]
        if self.patch_orders is None:
            self.patch_orders = [
                "column_major" if layer_type == "sliding_attention" and i % 9 >= 5 else "row_major"
                for i, layer_type in enumerate(self.layer_types)
            ]
        PreTrainedConfig.__post_init__(self, **kwargs)

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        if not set(self.layer_types) <= {"full_attention", "sliding_attention"}:
            raise ValueError("`layer_types` must only contain `'full_attention'` and `'sliding_attention'`.")
        if len(self.patch_orders) != self.depth or not set(self.patch_orders) <= {"row_major", "column_major"}:
            raise ValueError("`patch_orders` must contain `'row_major'` or `'column_major'` for every block.")
        if self.num_heads % self.num_key_value_heads != 0:
            raise ValueError("`num_heads` must be divisible by `num_key_value_heads`.")
        if self.head_dim % 4 != 0:
            raise ValueError("`head_dim` must be divisible by 4 for the 2D rotary embedding.")


@auto_docstring(checkpoint="XiaomiMiMo/MiMo-V2.6-Flash-RL")
@strict
class MiMoV2VLAudioConfig(PreTrainedConfig):
    r"""
    num_codebooks (`int`, *optional*, defaults to 20):
        Number of codebooks (residual vector quantization stages of the audio tokenizer) consumed by the model.
    group_size (`int`, *optional*, defaults to 4):
        Number of consecutive codec frames represented by one language-model token.
    head_dim (`int`, *optional*, defaults to 64):
        Dimension of the attention heads.
    out_hidden_size (`int`, *optional*, defaults to 4096):
        Output dimension of the audio projection; must match the language model's hidden size.
    codebook_vocab_sizes (`list[int]`, *optional*):
        Vocabulary size of the embedding of each codebook. Defaults to 1280 for every codebook.
    codebook_pad_token_ids (`list[int]`, *optional*):
        Padding codec ID of each codebook. Defaults to 1024 for every codebook.
    """

    model_type = "mimo_v2_vl_audio"
    base_config_key = "audio_config"
    default_theta = 640000.0
    # The transformer runs over the `group_size` codec frames of a single group.
    attribute_map = {"max_position_embeddings": "group_size"}

    num_codebooks: int = 20
    group_size: int = 4
    hidden_size: int = 1024
    num_hidden_layers: int = 6
    num_attention_heads: int = 16
    num_key_value_heads: int | None = None
    head_dim: int = 64
    intermediate_size: int = 4096
    hidden_act: str = "silu"
    attention_dropout: float = 0.0
    rms_norm_eps: float = 1e-6
    out_hidden_size: int = 4096
    codebook_vocab_sizes: list[int] | None = None
    codebook_pad_token_ids: list[int] | None = None
    rope_parameters: dict | None = None
    initializer_range: float = 0.02

    def __post_init__(self, **kwargs):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.codebook_vocab_sizes is None:
            self.codebook_vocab_sizes = [1280] * self.num_codebooks
        if self.codebook_pad_token_ids is None:
            self.codebook_pad_token_ids = [1024] * self.num_codebooks
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        for name in ("codebook_vocab_sizes", "codebook_pad_token_ids"):
            if len(getattr(self, name)) != self.num_codebooks:
                raise ValueError(f"`{name}` must contain one value per codebook ({self.num_codebooks}).")


@auto_docstring(checkpoint="XiaomiMiMo/MiMo-V2.6-Flash-RL")
@strict
class MiMoV2VLAudioTokenizerConfig(GlmAsrEncoderConfig):
    r"""
    num_mel_bins (`int`, *optional*, defaults to 128):
        Number of mel bins of the input features.
    conv_kernel_size (`int`, *optional*, defaults to 3):
        Kernel size of the two input convolutions.
    conv_stride (`int`, *optional*, defaults to 2):
        Stride of the second input convolution.
    layer_types (`list[str]`, *optional*):
        Attention type of each layer: `"sliding_attention"` (the `sliding_window` previous frames) or
        `"full_attention"`. Defaults to alternating sliding and full attention layers.
    sliding_window (`int`, *optional*, defaults to 128):
        Number of previous frames visible to each frame in `"sliding_attention"` layers.
    skip_layer_idx (`int`, *optional*, defaults to 2):
        Index of the layer whose output is added to the output of the last layer.
    downsample_factor (`int`, *optional*, defaults to 2):
        Temporal downsampling factor applied after the encoder layers.
    codebook_sizes (`list[int]`, *optional*):
        Number of entries of the codebook of each residual vector quantization stage, i.e. of each output codebook.
        Defaults to `[1024, 1024, 256]` followed by 17 codebooks of 128 entries.
    segment_length (`int`, *optional*, defaults to 6000):
        Maximum number of mel frames encoded at once; longer audios are split into segments.
    """

    model_type = "mimo_v2_vl_audio_tokenizer"
    base_config_key = "audio_tokenizer_config"
    default_theta = 10000.0
    # The encoder runs over segments of at most `segment_length` mel frames.
    attribute_map = {"max_position_embeddings": "segment_length"}

    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    conv_kernel_size: int = 3
    conv_stride: int = 2
    layer_norm_eps: float = 1e-5
    layer_types: list[str] | None = None
    sliding_window: int = 128
    skip_layer_idx: int = 2
    downsample_factor: int = 2
    codebook_sizes: list[int] | None = None
    segment_length: int = 6000
    max_position_embeddings = AttributeError()

    def __post_init__(self, **kwargs):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention" if i % 2 == 0 else "full_attention" for i in range(self.num_hidden_layers)
            ]
        if self.codebook_sizes is None:
            self.codebook_sizes = [1024, 1024, 256] + [128] * 17
        # Skips the GLM-ASR defaults: the rotary embedding covers the full head dimension.
        PreTrainedConfig.__post_init__(self, **kwargs)

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("`hidden_size` must be divisible by `num_attention_heads`.")
        if not 0 <= self.skip_layer_idx < self.num_hidden_layers:
            raise ValueError("`skip_layer_idx` must be the index of one of the encoder layers.")
        if self.num_key_value_heads != self.num_attention_heads:
            raise ValueError("The audio tokenizer only supports multi-head attention (`num_key_value_heads`).")


@auto_docstring(checkpoint="XiaomiMiMo/MiMo-V2.6-Flash-RL")
@strict
class MiMoV2VLConfig(Qwen2_5_VLConfig):
    r"""
    audio_tokenizer_config (`dict` or [`MiMoV2VLAudioTokenizerConfig`], *optional*):
        Configuration of the learned audio tokenizer, which turns log-mel features into codec IDs.
    vision_start_token_id (`int`, *optional*, defaults to 151652):
        Token ID marking the start of an image or video-frame span.
    vision_end_token_id (`int`, *optional*, defaults to 151653):
        Token ID marking the end of an image or video-frame span.
    audio_start_token_id (`int`, *optional*, defaults to 151673):
        Token ID marking the start of an audio span.
    audio_end_token_id (`int`, *optional*, defaults to 151674):
        Token ID marking the end of an audio span.
    """

    model_type = "mimo_v2_vl"
    sub_configs_defaults = {
        "text_config": SubConfigSpec(config_class=MiMoV2VLTextConfig),
        "vision_config": SubConfigSpec(config_class=MiMoV2VLVisionConfig),
        "audio_config": SubConfigSpec(config_class=MiMoV2VLAudioConfig),
        "audio_tokenizer_config": SubConfigSpec(config_class=MiMoV2VLAudioTokenizerConfig),
    }

    audio_config: dict | PreTrainedConfig | None = None
    audio_tokenizer_config: dict | PreTrainedConfig | None = None
    audio_token_id: int = 151669
    audio_start_token_id: int = 151673
    audio_end_token_id: int = 151674
    initializer_range: float = 0.02
    pad_token_id: int | None = None
    eos_token_id: int | list[int] | None = None

    def __post_init__(self, **kwargs):
        # Skips the Qwen2.5-VL backward compatibility with flat and mislabeled configurations.
        PreTrainedConfig.__post_init__(self, **kwargs)
        if self.pad_token_id is None:
            self.pad_token_id = self.text_config.pad_token_id
        if self.eos_token_id is None:
            self.eos_token_id = self.text_config.eos_token_id

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        hidden_size = self.text_config.hidden_size
        if self.vision_config.out_hidden_size != hidden_size or self.audio_config.out_hidden_size != hidden_size:
            raise ValueError("Vision and audio `out_hidden_size` must match `text_config.hidden_size`.")
        if len(self.audio_tokenizer_config.codebook_sizes) < self.audio_config.num_codebooks:
            raise ValueError("The audio tokenizer must produce at least `audio_config.num_codebooks` codebooks.")


class MiMoV2VLRMSNorm(MiMoV2FlashRMSNorm):
    pass


def apply_rotary_pos_emb_fp32(
    query: torch.Tensor, key: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotates the rotary part of queries and keys in float32 and leaves their remaining channels untouched."""
    rotary_dim = cos.shape[-1]
    query_rotary, key_rotary = apply_rotary_pos_emb(
        query[..., :rotary_dim].float(), key[..., :rotary_dim].float(), cos, sin
    )
    query = torch.cat((query_rotary.to(query.dtype), query[..., rotary_dim:]), dim=-1)
    key = torch.cat((key_rotary.to(key.dtype), key[..., rotary_dim:]), dim=-1)
    return query, key


class MiMoV2VLRotaryEmbedding(MiMoV2FlashRotaryEmbedding):
    @torch.no_grad()
    def forward(self, x, position_ids, layer_type):
        # Unlike MiMo-V2-Flash, the cos/sin tables stay in float32: queries and keys are rotated in float32.
        inv_freq = getattr(self, f"{layer_type}_inv_freq")
        attention_scaling = getattr(self, f"{layer_type}_attention_scaling")
        freqs = position_ids[..., None].float() * inv_freq.to(device=x.device, dtype=torch.float)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos() * attention_scaling, emb.sin() * attention_scaling


class MiMoV2VLAttention(MiMoV2FlashAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, None]:
        input_shape = hidden_states.shape[:-1]
        qk_hidden_shape = (*input_shape, -1, self.head_dim)
        v_hidden_shape = (*input_shape, -1, self.v_head_dim)

        query_states = self.q_proj(hidden_states).view(qk_hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(qk_hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(v_hidden_shape).transpose(1, 2) * self.v_scale

        cos, sin = position_embeddings
        # Rotating in float32 and rounding once is more accurate than rotating in the model dtype.
        query_states, key_states = apply_rotary_pos_emb_fp32(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            s_aux=self.sinks,  # Optional learned attention sinks
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class MiMoV2VLDecoderLayer(MiMoV2FlashDecoderLayer):
    pass


@auto_docstring
class MiMoV2VLPreTrainedModel(MiMoV2FlashPreTrainedModel):
    config: MiMoV2VLConfig
    input_modalities = ("image", "video", "audio", "text")
    _supports_sdpa = False  # SDPA has no API for the learned attention sinks
    _supports_flash_attn = False
    _supports_flex_attn = False
    _supports_attention_backend = False
    _can_compile_fullgraph = False
    _compatible_flash_implementations = AttributeError()
    _no_split_modules = [
        "MiMoV2VLDecoderLayer",
        "MiMoV2VLVisionBlock",
        "MiMoV2VLAudioEncoderLayer",
        "MiMoV2VLAudioTokenizerEncoderLayer",
    ]
    _keep_in_fp32_modules_strict = ["quantizer", "e_score_correction_bias"]
    _can_record_outputs = {"hidden_states": MiMoV2VLDecoderLayer}

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, MiMoV2VLAudioCodebook):
            init.zeros_(module.embed)
        elif isinstance(module, MiMoV2VLVisionAttention) and module.sinks is not None:
            init.zeros_(module.sinks)


@auto_docstring
class MiMoV2VLTextModel(MiMoV2FlashModel):
    config: MiMoV2VLTextConfig
    input_modalities = ("text",)
    _keep_in_fp32_modules_strict = ["e_score_correction_bias"]


class MiMoV2VLVisionMLP(Qwen2_5_VLMLP):
    pass


class MiMoV2VLVisionPatchEmbed(Qwen2_5_VisionPatchEmbed):
    pass


class MiMoV2VLVisionRotaryEmbedding(Qwen2_5_VLVisionRotaryEmbedding):
    pass


class MiMoV2VLVisionAttention(nn.Module):
    """
    Grouped-query attention restricted to each image or video frame. `"sliding_attention"` blocks only attend to the
    `sliding_window` patches on each side of the query, and may have learned attention sinks.
    """

    def __init__(self, config: MiMoV2VLVisionConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = config.head_dim
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = 0.0
        self.is_causal = False

        self.is_sliding = config.layer_types[layer_idx] == "sliding_attention"
        # CODEPATH: MiMo V2.5 and V2.6 run blocks 0, 9, 18 and 27 over the full image or frame, and the other blocks
        # over a window of `sliding_window` patches on each side, with learned attention sinks.
        self.sliding_window = config.sliding_window if self.is_sliding else None

        self.qkv = nn.Linear(self.dim, (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim, bias=True)
        self.proj = nn.Linear(self.num_heads * self.head_dim, self.dim)
        # The sinks are one extra logit per head in the softmax denominator.
        self.sinks = nn.Parameter(torch.zeros(self.num_heads)) if self.is_sliding else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        query_size = self.num_heads * self.head_dim
        key_value_size = self.num_key_value_heads * self.head_dim
        query_states, key_states, value_states = self.qkv(hidden_states).split(
            [query_size, key_value_size, key_value_size], dim=-1
        )
        query_states = query_states.view(seq_length, self.num_heads, self.head_dim)
        key_states = key_states.view(seq_length, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(seq_length, self.num_key_value_heads, self.head_dim)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        # Process each image or video frame separately.
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        splits = [torch.split(tensor, lengths, dim=2) for tensor in (query_states, key_states, value_states)]
        attn_outputs, attention_masks = [], {}
        for query, key, value in zip(*splits):
            attention_mask = None
            if self.sliding_window is not None:
                # The frames of a video share their length, and thus their mask.
                length = query.shape[2]
                if length not in attention_masks:
                    positions = torch.arange(length, device=query.device)
                    outside = (positions[None, :] - positions[:, None]).abs() > self.sliding_window
                    attention_masks[length] = query.new_zeros(outside.shape).masked_fill(outside, -torch.inf)[
                        None, None
                    ]
                attention_mask = attention_masks[length]
            attn_output, _ = attention_interface(
                self,
                query,
                key,
                value,
                attention_mask=attention_mask,
                scaling=self.scaling,
                dropout=0.0 if not self.training else self.attention_dropout,
                is_causal=False,
                **kwargs,
            )
            attn_outputs.append(attn_output)
        attn_output = torch.cat(attn_outputs, dim=1)

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.proj(attn_output)
        return attn_output


class MiMoV2VLVisionBlock(Qwen2_5_VLVisionBlock):
    def __init__(self, config: MiMoV2VLVisionConfig, layer_idx: int) -> None:
        super().__init__(config)
        self.norm1 = MiMoV2VLRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm2 = MiMoV2VLRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = MiMoV2VLVisionAttention(config, layer_idx)
        self.mlp = MiMoV2VLVisionMLP(config, bias=True)


class MiMoV2VLVisionPatchMerger(Qwen3VLVisionPatchMerger):
    """
    Normalizes every patch with RMSNorm, groups the `spatial_merge_size**2` patches of each merge block, and projects
    them to the language model with a 2-layer MLP.
    """

    def __init__(self, config: MiMoV2VLVisionConfig) -> None:
        super().__init__(config)
        # No DeepStack mergers: the norm always applies to the patches before grouping.
        self.use_postshuffle_norm = False
        self.norm = MiMoV2VLRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.linear_fc2 = nn.Linear(self.hidden_size, config.out_hidden_size, bias=False)


def get_column_major_block_order(grid_thw: torch.Tensor, spatial_merge_size: int) -> torch.Tensor:
    """
    Returns the permutation of the `spatial_merge_size**2` patch blocks that traverses every image or video frame
    column by column, instead of the row-by-row order of the processor.
    """
    block_order, offset = [], 0
    for t, h, w in grid_thw.tolist():
        num_blocks = t * (h // spatial_merge_size) * (w // spatial_merge_size)
        blocks = torch.arange(num_blocks, device=grid_thw.device)
        blocks = blocks.view(t, h // spatial_merge_size, w // spatial_merge_size)
        block_order.append(blocks.transpose(1, 2).flatten() + offset)
        offset += num_blocks
    return torch.cat(block_order)


@auto_docstring
class MiMoV2VLVisionModel(MiMoV2VLPreTrainedModel):
    config: MiMoV2VLVisionConfig
    input_modalities = ("image", "video")
    _no_split_modules = ["MiMoV2VLVisionBlock"]
    _can_record_outputs = {"hidden_states": MiMoV2VLVisionBlock}

    def __init__(self, config: MiMoV2VLVisionConfig, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)
        self.spatial_merge_size = config.spatial_merge_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

        self.patch_embed = MiMoV2VLVisionPatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            embed_dim=config.hidden_size,
        )
        self.rotary_pos_emb = MiMoV2VLVisionRotaryEmbedding(config)
        self.blocks = nn.ModuleList([MiMoV2VLVisionBlock(config, layer_idx) for layer_idx in range(config.depth)])
        self.merger = MiMoV2VLVisionPatchMerger(config)

        self.gradient_checkpointing = False
        self.post_init()

    def permute_blocks(self, hidden_states: torch.Tensor, block_order: torch.Tensor) -> torch.Tensor:
        """Reorders the `spatial_merge_unit` patch blocks of `hidden_states` (first dimension) as in `block_order`."""
        hidden_states = hidden_states.unflatten(0, (-1, self.spatial_merge_unit))
        return hidden_states[block_order.to(hidden_states.device)].flatten(0, 1)

    @capture_outputs
    @auto_docstring
    def forward(
        self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs: Unpack[TransformersKwargs]
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        grid_thw (`torch.LongTensor` of shape `(num_images_or_videos, 3)`):
            The temporal, height and width patch grid of each image or video.
        """
        position_ids = get_vision_position_ids(grid_thw, self.spatial_merge_size, kwargs=kwargs)
        cu_seqlens, _ = get_vision_attention_seqlens(grid_thw, self.config, kwargs=kwargs)

        hidden_states = self.patch_embed(hidden_states)
        position_embeddings = self.rotary_pos_emb(hidden_states, position_ids)

        # `"column_major"` blocks traverse each frame column by column, which only permutes the patch blocks inside
        # every frame: `cu_seqlens` is unchanged.
        column_order = get_column_major_block_order(grid_thw, self.spatial_merge_size)
        row_order = column_order.argsort()
        column_position_embeddings = tuple(self.permute_blocks(emb, column_order) for emb in position_embeddings)

        in_column_order = False
        for block, patch_order in zip(self.blocks, self.config.patch_orders):
            use_column_order = patch_order == "column_major"
            # CODEPATH: in MiMo V2.5 and V2.6, the last 4 sliding blocks of every cycle of 9 blocks traverse the patches
            # column by column: permute them when the traversal order changes between consecutive blocks.
            if use_column_order != in_column_order:
                hidden_states = self.permute_blocks(hidden_states, column_order if use_column_order else row_order)
                in_column_order = use_column_order
            hidden_states = block(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=column_position_embeddings if use_column_order else position_embeddings,
                **kwargs,
            )
        if in_column_order:
            hidden_states = self.permute_blocks(hidden_states, row_order)

        merged_hidden_states = self.merger(hidden_states)
        return BaseModelOutputWithPooling(last_hidden_state=hidden_states, pooler_output=merged_hidden_states)


class MiMoV2VLAudioRotaryEmbedding(Qwen2RotaryEmbedding):
    pass


class MiMoV2VLAudioMLP(Qwen2MLP):
    pass


class MiMoV2VLAudioAttention(Qwen2Attention):
    """Bidirectional attention between the codec frames of a single group."""

    def __init__(self, config: MiMoV2VLAudioConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.is_causal = False
        # Read by the eager attention of the language model, which supports learned attention sinks.
        self.sinks = None


class MiMoV2VLAudioEncoderLayer(Qwen2DecoderLayer):
    def __init__(self, config: MiMoV2VLAudioConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = MiMoV2VLAudioAttention(config, layer_idx)
        self.mlp = MiMoV2VLAudioMLP(config)
        self.input_layernorm = MiMoV2VLRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MiMoV2VLRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class MiMoV2VLAudioProjection(Qwen3OmniMoeTalkerResizeMLP):
    def __init__(self, config: MiMoV2VLAudioConfig):
        nn.Module.__init__(self)
        input_size = config.hidden_size * config.group_size
        self.linear_fc1 = nn.Linear(input_size, input_size * 4, bias=False)
        self.linear_fc2 = nn.Linear(input_size * 4, config.out_hidden_size, bias=False)
        self.act_fn = nn.GELU()


@auto_docstring
class MiMoV2VLAudioEncoder(MiMoV2VLPreTrainedModel):
    """
    Embeds the codec IDs of every group of `group_size` codec frames (summing the embeddings of all the codebooks),
    contextualizes the frames of each group with a small transformer, and projects each group to one language-model
    embedding.
    """

    config: MiMoV2VLAudioConfig
    input_modalities = "audio"
    _no_split_modules = ["MiMoV2VLAudioEncoderLayer"]
    _can_record_outputs = {"hidden_states": MiMoV2VLAudioEncoderLayer}

    def __init__(self, config: MiMoV2VLAudioConfig):
        super().__init__(config)
        self.embed_tokens = nn.ModuleList(
            [
                nn.Embedding(vocab_size, config.hidden_size, padding_idx=pad_token_id)
                for vocab_size, pad_token_id in zip(config.codebook_vocab_sizes, config.codebook_pad_token_ids)
            ]
        )
        self.layers = nn.ModuleList(
            [MiMoV2VLAudioEncoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = MiMoV2VLRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = MiMoV2VLAudioRotaryEmbedding(config)
        self.projection = MiMoV2VLAudioProjection(config)
        self.post_init()

    @capture_outputs
    @auto_docstring
    def forward(
        self, input_audio_codes: torch.LongTensor, **kwargs: Unpack[TransformersKwargs]
    ) -> BaseModelOutputWithPooling:
        r"""
        input_audio_codes (`torch.LongTensor` of shape `(num_groups, group_size, num_codebooks)`):
            Codec IDs of each group of codec frames.
        """
        hidden_states = self.embed_tokens[0](input_audio_codes[..., 0])
        for codebook, embed_tokens in enumerate(self.embed_tokens[1:], start=1):
            hidden_states = hidden_states + embed_tokens(input_audio_codes[..., codebook]).to(hidden_states.device)

        position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device)[None]
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states, attention_mask=None, position_embeddings=position_embeddings, **kwargs
            )
        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPooling(
            last_hidden_state=hidden_states, pooler_output=self.projection(hidden_states.flatten(1))
        )


class MiMoV2VLAudioTokenizerRotaryEmbedding(OlmoRotaryEmbedding):
    # Unlike the audio encoder, the cos/sin tables stay in float32: queries and keys are rotated in float32.
    pass


class MiMoV2VLAudioCodebook(EncodecEuclideanCodebook):
    """
    Euclidean codebook of one residual vector quantization stage. Only the codebook entries are kept: the training
    statistics of the original checkpoints (`inited`, `cluster_size`, `embed_avg`) are not used for encoding.
    """

    def __init__(self, hidden_size: int, codebook_size: int):
        nn.Module.__init__(self)
        self.codebook_size = codebook_size
        self.embed = nn.Buffer(torch.zeros(codebook_size, hidden_size))

    def quantize(self, hidden_states):
        # The distances are computed in float32, even when the model was cast to a lower precision after loading.
        embed = self.embed.float().t()
        scaled_states = hidden_states.pow(2).sum(1, keepdim=True)
        dist = -(scaled_states - 2 * hidden_states @ embed + embed.pow(2).sum(0, keepdim=True))
        return dist.max(dim=-1).indices


class MiMoV2VLAudioVectorQuantization(EncodecVectorQuantization):
    def __init__(self, hidden_size: int, codebook_size: int):
        nn.Module.__init__(self)
        self.codebook = MiMoV2VLAudioCodebook(hidden_size, codebook_size)


class MiMoV2VLAudioResidualVectorQuantizer(nn.Module):
    """Residual vector quantizer whose stages have codebooks of different sizes."""

    def __init__(self, config: MiMoV2VLAudioTokenizerConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                MiMoV2VLAudioVectorQuantization(config.hidden_size, codebook_size)
                for codebook_size in config.codebook_sizes
            ]
        )

    def encode(self, hidden_states: torch.Tensor) -> torch.LongTensor:
        """Returns the codec IDs of shape `(num_frames, num_codebooks)` of `hidden_states` (`(num_frames, hidden_size)`)."""
        # The quantization stages take `(batch_size, hidden_size, time)` inputs, and run in float32 like their codebooks.
        residual = hidden_states.float().T[None]
        codes = []
        for layer in self.layers:
            indices = layer.encode(residual)
            residual = residual - layer.decode(indices)
            codes.append(indices[0])
        return torch.stack(codes, dim=-1)


class MiMoV2VLAudioTokenizerAttention(GlmAsrAttention):
    def __init__(self, config: MiMoV2VLAudioTokenizerConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        # Causality is enforced by the attention masks of the encoder.
        self.is_causal = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_fp32(query_states, key_states, cos, sin)

        # Like the original implementation, the masked attention always runs with SDPA.
        attn_output = F.scaled_dot_product_attention(
            query_states, key_states, value_states, attn_mask=attention_mask, scale=self.scaling
        )
        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output), None


class MiMoV2VLAudioTokenizerMLP(GlmAsrMLP):
    pass


class MiMoV2VLAudioTokenizerEncoderLayer(GlmAsrEncoderLayer):
    def __init__(self, config: MiMoV2VLAudioTokenizerConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = MiMoV2VLAudioTokenizerAttention(config=config, layer_idx=layer_idx)
        self.mlp = MiMoV2VLAudioTokenizerMLP(config)
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)


@auto_docstring
@dataclass
class MiMoV2VLAudioTokenizerOutput(ModelOutput):
    r"""
    audio_codes (`list[torch.LongTensor]`):
        Codec IDs of shape `(codec_time, num_codebooks)` of each audio.
    """

    audio_codes: list[torch.LongTensor] | None = None


@auto_docstring
class MiMoV2VLAudioTokenizerEncoder(MiMoV2VLPreTrainedModel):
    """
    Learned audio tokenizer turning log-mel features into codec IDs: two convolutions, a transformer encoder, a
    temporal downsampling and a residual vector quantizer. It is frozen, as only its encoding is used.
    """

    config: MiMoV2VLAudioTokenizerConfig
    input_modalities = "audio"
    _no_split_modules = ["MiMoV2VLAudioTokenizerEncoderLayer"]

    def __init__(self, config: MiMoV2VLAudioTokenizerConfig):
        super().__init__(config)
        self.conv1 = nn.Conv1d(config.num_mel_bins, config.hidden_size, kernel_size=config.conv_kernel_size, padding=1)
        self.conv2 = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=config.conv_kernel_size,
            stride=config.conv_stride,
            padding=1,
        )
        self.rotary_emb = MiMoV2VLAudioTokenizerRotaryEmbedding(config)
        self.layers = nn.ModuleList(
            [MiMoV2VLAudioTokenizerEncoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.downsample_conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=config.downsample_factor,
            stride=config.downsample_factor,
            bias=False,
        )
        self.downsample_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.quantizer = MiMoV2VLAudioResidualVectorQuantizer(config)
        self.post_init()

    def _get_conv_output_lengths(self, lengths: torch.LongTensor) -> torch.LongTensor:
        kernel_size, stride = self.config.conv_kernel_size, self.config.conv_stride
        lengths = lengths + 2 - kernel_size + 1  # conv1: padding 1, stride 1
        return (lengths + 2 - kernel_size) // stride + 1  # conv2: padding 1

    def _get_attention_masks(self, lengths: torch.LongTensor, sequence_length: int) -> dict:
        """Returns the causal boolean attention mask of each layer type, hiding the padding."""
        positions = torch.arange(sequence_length, device=lengths.device)
        distance = positions[None, :] - positions[:, None]
        allowed = (positions[None, None, :] < lengths[:, None, None]) & (distance <= 0)
        masks = {"full_attention": allowed, "sliding_attention": allowed & (distance >= -self.config.sliding_window)}
        return {layer_type: mask[:, None] for layer_type, mask in masks.items()}

    @torch.no_grad()
    def encode_segments(self, input_features: torch.Tensor, lengths: torch.LongTensor) -> list[torch.LongTensor]:
        """Encodes a padded batch of mel segments into one `(codec_time, num_codebooks)` tensor per segment."""
        input_features = input_features.to(dtype=self.conv1.weight.dtype)
        hidden_states = F.gelu(self.conv1(input_features.transpose(1, 2)))
        hidden_states = F.gelu(self.conv2(hidden_states)).transpose(1, 2)
        lengths = self._get_conv_output_lengths(lengths.to(hidden_states.device))

        position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device)[None]
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        attention_masks = self._get_attention_masks(lengths, hidden_states.shape[1])
        for layer_idx, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_masks[self.config.layer_types[layer_idx]],
            )
            # CODEPATH: MiMo V2.5 and V2.6 add the output of their third layer to the output of the last layer.
            if layer_idx == self.config.skip_layer_idx:
                skip_hidden_states = hidden_states
        hidden_states = hidden_states + skip_hidden_states.to(hidden_states.device)
        hidden_states = self.layer_norm(hidden_states)
        lengths = lengths.to(hidden_states.device)

        # Like the original implementation, padded frames repeat the last valid frame before downsampling.
        downsample_factor = self.config.downsample_factor
        valid = torch.arange(hidden_states.shape[1], device=lengths.device)[None] < lengths[:, None]
        last_hidden = hidden_states[torch.arange(len(lengths), device=lengths.device), lengths - 1, None]
        hidden_states = torch.where(valid[..., None], hidden_states, last_hidden)
        if pad_length := -hidden_states.shape[1] % downsample_factor:
            hidden_states = F.pad(hidden_states, (0, 0, 0, pad_length))
        hidden_states = F.gelu(self.downsample_conv(hidden_states.transpose(1, 2))).transpose(1, 2)
        hidden_states = self.downsample_norm(hidden_states)
        lengths = (lengths + downsample_factor - 1) // downsample_factor

        valid = torch.arange(hidden_states.shape[1], device=lengths.device)[None] < lengths[:, None]
        codes = self.quantizer.encode(hidden_states[valid])
        return list(codes.split(lengths.tolist(), dim=0))

    @torch.no_grad()
    @auto_docstring
    def forward(
        self,
        input_features: torch.Tensor,
        feature_attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MiMoV2VLAudioTokenizerOutput:
        r"""
        input_features (`torch.FloatTensor` of shape `(batch_size, mel_time, num_mel_bins)`):
            Log-mel features of the audios. Audios longer than `segment_length` mel frames are encoded in segments.
        feature_attention_mask (`torch.Tensor` of shape `(batch_size, mel_time)`, *optional*):
            Mask of the valid mel frames of each audio.
        """
        if feature_attention_mask is None:
            lengths = [input_features.shape[1]] * len(input_features)
        else:
            # trf-ignore: TRF056 — the audios are split into segments of `segment_length` frames on the host.
            lengths = feature_attention_mask.sum(-1, dtype=torch.long).tolist()

        segments, num_segments = [], []
        for index, (features, length) in enumerate(zip(input_features, lengths)):
            if feature_attention_mask is not None:
                features = features[feature_attention_mask[index].to(device=features.device, dtype=torch.bool)]
            audio_segments = features[:length].split(self.config.segment_length, dim=0)
            segments.extend(audio_segments)
            num_segments.append(len(audio_segments))

        segment_lengths = torch.tensor([len(segment) for segment in segments], device=input_features.device)
        padded_segments = nn.utils.rnn.pad_sequence(segments, batch_first=True)
        segment_codes = iter(self.encode_segments(padded_segments, segment_lengths))
        audio_codes = [torch.cat([next(segment_codes) for _ in range(count)]) for count in num_segments]
        return MiMoV2VLAudioTokenizerOutput(audio_codes=audio_codes)


_AUDIO_ARGS_DOCSTRING = r"""
input_audio_codes (`torch.LongTensor` of shape `(batch_size, grouped_time, group_size, num_codebooks)`, *optional*):
    Grouped codec IDs of pre-tokenized audio. Mutually exclusive with `input_features`.
feature_attention_mask (`torch.Tensor` of shape `(batch_size, mel_time)`, *optional*):
    Mask identifying valid mel frames in `input_features`.
audio_feature_lens (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
    Number of audio tokens of each audio, before batch padding.
audio_segment_offsets (`torch.LongTensor` of shape `(num_segments, 3)`, *optional*):
    `(audio_index, start, length)` rows selecting, in prompt order, the audio tokens that fill each audio span.
"""


@auto_docstring
class MiMoV2VLModel(Qwen2_5_VLModel):
    # Also keeps the audio layers of `MiMoV2VLPreTrainedModel` unsplit.
    _no_split_modules = AttributeError()
    # The loss is normalized with `num_items_in_batch` when training with gradient accumulation.
    accepts_loss_kwargs = AttributeError()

    def __init__(self, config: MiMoV2VLConfig):
        MiMoV2VLPreTrainedModel.__init__(self, config)
        self.language_model = MiMoV2VLTextModel(config.text_config)
        self.visual = MiMoV2VLVisionModel(config.vision_config)
        self.audio_encoder = MiMoV2VLAudioEncoder(config.audio_config)
        self.audio_tokenizer = MiMoV2VLAudioTokenizerEncoder(config.audio_tokenizer_config)
        self.post_init()

    def get_vision_position_ids(self, *args, **kwargs):
        raise AttributeError("MiMo V2 VL uses 1D position IDs for all modalities.")

    def get_rope_index(self, *args, **kwargs):
        raise AttributeError("MiMo V2 VL does not use 3D rotary position IDs.")

    def compute_3d_position_ids(self, *args, **kwargs):
        raise AttributeError("MiMo V2 VL does not use 3D rotary position IDs.")

    @accepts_precomputed_kwargs(modality="image")
    @can_return_tuple
    @auto_docstring
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        pixel_values (`torch.FloatTensor` of shape `(num_patches, patch_dim)`):
            Flattened patches of the images, as returned by [`MiMoV2VLImageProcessor`], with
            `patch_dim = num_channels * temporal_patch_size * patch_size * patch_size`.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`):
            The temporal, height and width patch grid of each image.
        """
        pixel_values = pixel_values.type(self.visual.dtype)
        vision_outputs = self.visual(pixel_values, grid_thw=image_grid_thw, return_dict=True, **kwargs)
        split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        vision_outputs.pooler_output = torch.split(vision_outputs.pooler_output, split_sizes)
        return vision_outputs

    @accepts_precomputed_kwargs(modality="video")
    @can_return_tuple
    @auto_docstring
    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_grid_thw: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        pixel_values_videos (`torch.FloatTensor` of shape `(num_patches, patch_dim)`):
            Flattened patches of the videos, as returned by [`MiMoV2VLVideoProcessor`], with
            `patch_dim = num_channels * temporal_patch_size * patch_size * patch_size`.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`):
            The temporal, height and width patch grid of each video.
        """
        pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
        vision_outputs = self.visual(pixel_values_videos, grid_thw=video_grid_thw, return_dict=True, **kwargs)
        split_sizes = (video_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        vision_outputs.pooler_output = torch.split(vision_outputs.pooler_output, split_sizes)
        return vision_outputs

    def get_audio_codes(
        self,
        input_audio_codes: torch.LongTensor | None = None,
        input_features: torch.FloatTensor | None = None,
        feature_attention_mask: torch.Tensor | None = None,
    ) -> list[torch.LongTensor]:
        """Returns one `(grouped_time, group_size, num_codebooks)` tensor of codec IDs per audio."""
        if (input_audio_codes is None) == (input_features is None):
            raise ValueError("You must specify exactly one of `input_features` or `input_audio_codes`.")
        if input_audio_codes is not None:
            return list(input_audio_codes)

        group_size, num_codebooks = self.config.audio_config.group_size, self.config.audio_config.num_codebooks
        grouped_codes = []
        for codes in self.audio_tokenizer(input_features, feature_attention_mask).audio_codes:
            # Complete the last group by repeating the final codec frame.
            if pad_length := -len(codes) % group_size:
                codes = torch.cat((codes, codes[-1:].expand(pad_length, -1)))
            grouped_codes.append(codes[:, :num_codebooks].reshape(-1, group_size, num_codebooks))
        return grouped_codes

    @can_return_tuple
    @auto_docstring(custom_args=_AUDIO_ARGS_DOCSTRING)
    def get_audio_features(
        self,
        input_audio_codes: torch.LongTensor | None = None,
        audio_feature_lens: torch.LongTensor | None = None,
        audio_segment_offsets: torch.LongTensor | None = None,
        input_features: torch.FloatTensor | None = None,
        feature_attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        grouped_codes = self.get_audio_codes(input_audio_codes, input_features, feature_attention_mask)
        lengths = [len(codes) for codes in grouped_codes]
        if audio_feature_lens is not None:
            lengths = torch.as_tensor(audio_feature_lens).tolist()
            if len(lengths) != len(grouped_codes) or any(
                not 0 <= length <= len(codes) for length, codes in zip(lengths, grouped_codes)
            ):
                raise ValueError("`audio_feature_lens` does not match the encoded audio sequences.")

        # By default, each audio fills one span; video audio tracks are split across interleaved spans.
        if audio_segment_offsets is None:
            audio_segment_offsets = [(index, 0, length) for index, length in enumerate(lengths)]
        segments = []
        for index, start, length in torch.as_tensor(audio_segment_offsets).tolist():
            if start + length > lengths[index]:
                raise ValueError("`audio_segment_offsets` extend beyond the valid codec sequence.")
            segments.append(grouped_codes[index][start : start + length])
        return self.audio_encoder(torch.cat(segments).long(), return_dict=True, **kwargs)

    def get_placeholder_mask(
        self, input_ids: torch.LongTensor, inputs_embeds: torch.FloatTensor, token_id: int, features: torch.Tensor
    ) -> torch.BoolTensor:
        """
        Obtains the placeholder mask of `token_id` from `input_ids` or `inputs_embeds`, and checks that the number of
        placeholder tokens is equal to the number of multimodal features.
        """
        if input_ids is None:
            placeholder_embedding = self.get_input_embeddings()(
                torch.tensor(token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_mask = (inputs_embeds == placeholder_embedding).all(-1)
        else:
            special_mask = input_ids.to(inputs_embeds.device) == token_id
        num_tokens = special_mask.sum()
        torch_compilable_check(
            num_tokens == features.shape[0],
            f"Multimodal features and placeholder tokens do not match: tokens: {num_tokens}, "
            f"features: {features.shape[0]}",
        )
        return special_mask.unsqueeze(-1).expand_as(inputs_embeds)

    def get_multimodal_embeddings(
        self,
        input_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        input_audio_codes: torch.LongTensor | None = None,
        input_features: torch.FloatTensor | None = None,
        feature_attention_mask: torch.Tensor | None = None,
        audio_feature_lens: torch.LongTensor | None = None,
        audio_segment_offsets: torch.LongTensor | None = None,
        mm_encoder_outputs: dict[str, BaseModelOutputWithPooling] | None = None,
    ) -> torch.FloatTensor:
        """Embeds `input_ids` and replaces the image, video and audio placeholders by their encoder features."""
        if (pixel_values is not None or pixel_values_videos is not None) and mm_encoder_outputs is not None:
            raise ValueError(
                "You cannot specify both pixel_values/pixel_values_videos and mm_encoder_outputs at the same time"
            )
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        mm_encoder_outputs = dict(mm_encoder_outputs or {})
        if pixel_values is not None:
            mm_encoder_outputs["image"] = self.get_image_features(pixel_values, image_grid_thw, return_dict=True)
        if pixel_values_videos is not None:
            mm_encoder_outputs["video"] = self.get_video_features(
                pixel_values_videos, video_grid_thw, return_dict=True
            )

        modality_features = {
            modality: torch.cat(outputs.pooler_output)
            for modality, outputs in mm_encoder_outputs.items()
            if outputs is not None
        }
        if input_audio_codes is not None or input_features is not None:
            modality_features["audio"] = self.get_audio_features(
                input_audio_codes=input_audio_codes,
                audio_feature_lens=audio_feature_lens,
                audio_segment_offsets=audio_segment_offsets,
                input_features=input_features,
                feature_attention_mask=feature_attention_mask,
                return_dict=True,
            ).pooler_output

        for modality, features in modality_features.items():
            token_id = getattr(self.config, f"{modality}_token_id")
            special_mask = self.get_placeholder_mask(input_ids, inputs_embeds, token_id, features)
            features = features.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(special_mask, features)
        return inputs_embeds

    @can_return_tuple
    @auto_docstring(custom_args=_AUDIO_ARGS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        input_audio_codes: torch.LongTensor | None = None,
        input_features: torch.FloatTensor | None = None,
        feature_attention_mask: torch.Tensor | None = None,
        audio_feature_lens: torch.LongTensor | None = None,
        audio_segment_offsets: torch.LongTensor | None = None,
        mm_encoder_outputs: dict[str, BaseModelOutputWithPooling] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPast:
        r"""
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence token. Image, video and audio tokens use the same 1D positions
            as the text tokens.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        inputs_embeds = self.get_multimodal_embeddings(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            input_audio_codes=input_audio_codes,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            audio_feature_lens=audio_feature_lens,
            audio_segment_offsets=audio_segment_offsets,
            mm_encoder_outputs=mm_encoder_outputs,
        )
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=True,
            **kwargs,
        )
        return BaseModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# Inputs that are only consumed while prefilling the cache.
_MULTIMODAL_INPUT_NAMES = (
    "pixel_values",
    "image_grid_thw",
    "pixel_values_videos",
    "video_grid_thw",
    "input_audio_codes",
    "input_features",
    "feature_attention_mask",
    "audio_feature_lens",
    "audio_segment_offsets",
    "mm_encoder_outputs",
)


@auto_docstring
class MiMoV2VLForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    accepts_loss_kwargs = AttributeError()

    @auto_docstring
    def get_image_features(self, **super_kwargs):
        r"""
        pixel_values (`torch.FloatTensor` of shape `(num_patches, patch_dim)`):
            Flattened patches of the images, as returned by [`MiMoV2VLImageProcessor`], with
            `patch_dim = num_channels * temporal_patch_size * patch_size * patch_size`.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`):
            The temporal, height and width patch grid of each image.
        """
        return super().get_image_features(**super_kwargs)

    @auto_docstring
    def get_video_features(self, **super_kwargs):
        r"""
        pixel_values_videos (`torch.FloatTensor` of shape `(num_patches, patch_dim)`):
            Flattened patches of the videos, as returned by [`MiMoV2VLVideoProcessor`], with
            `patch_dim = num_channels * temporal_patch_size * patch_size * patch_size`.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`):
            The temporal, height and width patch grid of each video.
        """
        return super().get_video_features(**super_kwargs)

    @auto_docstring(custom_args=_AUDIO_ARGS_DOCSTRING)
    def get_audio_features(
        self,
        input_audio_codes: torch.LongTensor | None = None,
        audio_feature_lens: torch.LongTensor | None = None,
        audio_segment_offsets: torch.LongTensor | None = None,
        input_features: torch.FloatTensor | None = None,
        feature_attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        return self.model.get_audio_features(
            input_audio_codes=input_audio_codes,
            audio_feature_lens=audio_feature_lens,
            audio_segment_offsets=audio_segment_offsets,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            **kwargs,
        )

    @can_return_tuple
    @auto_docstring(custom_args=_AUDIO_ARGS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        input_audio_codes: torch.LongTensor | None = None,
        input_features: torch.FloatTensor | None = None,
        feature_attention_mask: torch.Tensor | None = None,
        audio_feature_lens: torch.LongTensor | None = None,
        audio_segment_offsets: torch.LongTensor | None = None,
        mm_encoder_outputs: dict[str, BaseModelOutputWithPooling] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.text_config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are
            ignored (masked), the loss is only computed for the tokens with labels in `[0, ...,
            config.text_config.vocab_size]`.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            input_audio_codes=input_audio_codes,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            audio_feature_lens=audio_feature_lens,
            audio_segment_offsets=audio_segment_offsets,
            mm_encoder_outputs=mm_encoder_outputs,
            return_dict=True,
            **kwargs,
        )

        hidden_states = slice_logits_to_keep(outputs.last_hidden_state, logits_to_keep)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, is_first_iteration=False, **kwargs):
        use_cache = kwargs.get("use_cache", True)
        past_key_values = kwargs.get("past_key_values")
        # Media is encoded when prefilling an empty cache and is not needed by the following decoding steps.
        is_first_iteration = is_first_iteration or (
            use_cache and (past_key_values is None or past_key_values.get_seq_length() == 0)
        )
        model_inputs = GenerationMixin.prepare_inputs_for_generation(
            self, input_ids, is_first_iteration=is_first_iteration, **kwargs
        )
        if use_cache and not is_first_iteration:
            for name in _MULTIMODAL_INPUT_NAMES:
                model_inputs.pop(name, None)
        return model_inputs

    def _prepare_position_ids_for_generation(self, *args, **kwargs):
        raise AttributeError("MiMo V2 VL uses the default 1D position IDs.")

    def _expand_inputs_for_generation(self, expand_size=1, is_encoder_decoder=False, input_ids=None, **model_kwargs):
        has_audio = model_kwargs.get("input_audio_codes") is not None or model_kwargs.get("input_features") is not None
        if expand_size > 1 and has_audio:
            # `audio_segment_offsets` index whole audios, which cannot be repeated per sequence: merge all media into
            # `inputs_embeds` first. Images and videos alone are pre-encoded in `mm_encoder_outputs` by `generate`.
            multimodal_inputs = {
                name: model_kwargs.pop(name) for name in _MULTIMODAL_INPUT_NAMES if name in model_kwargs
            }
            model_kwargs["inputs_embeds"] = self.model.get_multimodal_embeddings(
                input_ids, inputs_embeds=model_kwargs.get("inputs_embeds"), **multimodal_inputs
            )
        return GenerationMixin._expand_inputs_for_generation(
            self, expand_size=expand_size, is_encoder_decoder=is_encoder_decoder, input_ids=input_ids, **model_kwargs
        )


def smart_resize(height: int, width: int, factor: int, min_pixels: int, max_pixels: int) -> tuple[int, int]:
    """Applies MiMo's minimum merge-block scaling before Qwen-style pixel-budget rounding."""
    if min(height, width, factor, min_pixels) <= 0 or max_pixels < min_pixels:
        raise ValueError("Image dimensions and pixel limits must be positive and min_pixels <= max_pixels.")
    if min(height, width) < factor:
        scale = factor / min(height, width)
        height, width = round(height * scale), round(width * scale)
    elif max(height, width) / min(height, width) > 200:
        raise ValueError("Image aspect ratio must not exceed 200.")
    resized_height = round(height / factor) * factor
    resized_width = round(width / factor) * factor
    if resized_height * resized_width > max_pixels:
        scale = math.sqrt(height * width / max_pixels)
        resized_height = math.floor(height / scale / factor) * factor
        resized_width = math.floor(width / scale / factor) * factor
    elif resized_height * resized_width < min_pixels:
        scale = math.sqrt(min_pixels / (height * width))
        resized_height = math.ceil(height * scale / factor) * factor
        resized_width = math.ceil(width * scale / factor) * factor
    if min(resized_height, resized_width) == 0:
        raise ValueError("Pixel budget is too small for the image aspect ratio.")
    return resized_height, resized_width


def resize_frames(
    processor: TorchvisionBackend,
    frames: torch.Tensor,
    size: SizeDict,
    resample: PILImageResampling,
    factor: int,
) -> torch.Tensor:
    """Resizes the frames in float32 without antialiasing, as in the original MiMo processor."""
    if not size.shortest_edge or not size.longest_edge:
        raise ValueError("size must contain shortest_edge and longest_edge pixel budgets.")
    height, width = smart_resize(
        *frames.shape[-2:], factor=factor, min_pixels=size.shortest_edge, max_pixels=size.longest_edge
    )
    frames = frames.float()
    if tuple(frames.shape[-2:]) == (height, width):
        return frames
    return TorchvisionBackend.resize(
        processor, frames, SizeDict(height=height, width=width), resample=resample, antialias=False
    )


class MiMoV2VLImageProcessorKwargs(Qwen2VLImageProcessorKwargs, total=False):
    """
    min_pixels (`int`, *optional*, defaults to `8 * 32 * 32`):
        Minimum image pixel budget. Prefer `size["shortest_edge"]`.
    max_pixels (`int`, *optional*, defaults to `8192 * 32 * 32`):
        Maximum image pixel budget. Prefer `size["longest_edge"]`.
    patch_size (`int`, *optional*, defaults to 16):
        Spatial patch size of the vision encoder.
    temporal_patch_size (`int`, *optional*, defaults to 2):
        Number of repeated image frames in each temporal patch.
    merge_size (`int`, *optional*, defaults to 2):
        Number of spatial patches merged along each axis.
    """


@auto_docstring(custom_intro="Constructs an image processor for the MiMo V2 VL series.")
class MiMoV2VLImageProcessor(Qwen2VLImageProcessor):
    size = {"shortest_edge": 8 * 32 * 32, "longest_edge": 8192 * 32 * 32}
    resample = PILImageResampling.BILINEAR
    patch_size = 16
    valid_kwargs = MiMoV2VLImageProcessorKwargs

    def process_image(
        self,
        image: ImageInput,
        do_convert_rgb: bool | None = None,
        input_data_format: str | ChannelDimension | None = None,
        device: str | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if isinstance(image, np.ndarray) and any(stride < 0 for stride in image.strides):
            image = np.ascontiguousarray(image)
        return TorchvisionBackend.process_image(
            self, image, do_convert_rgb=do_convert_rgb, input_data_format=input_data_format, device=device, **kwargs
        )

    def resize(
        self, images: torch.Tensor, size: SizeDict, resample: PILImageResampling, factor: int, **kwargs
    ) -> torch.Tensor:
        return resize_frames(self, images, size, resample, factor)

    def patchify(
        self, images: torch.Tensor, patch_size: int, merge_size: int, temporal_patch_size: int
    ) -> tuple[torch.Tensor, int, int]:
        if min(patch_size, temporal_patch_size, merge_size) <= 0:
            raise ValueError("Patch and merge sizes must be positive.")
        if any(length % (patch_size * merge_size) for length in images.shape[-2:]):
            raise ValueError("Frame dimensions must be divisible by their temporal and spatial patch factors.")
        # The patches are contiguous, as the patch embedding of the vision encoder reshapes them with `view`.
        return super().patchify(images, patch_size, merge_size, temporal_patch_size)

    def get_number_of_image_patches(self, height: int, width: int, images_kwargs: dict | None = None) -> int:
        images_kwargs = images_kwargs or {}
        patch_size = images_kwargs.get("patch_size", self.patch_size)
        merge_size = images_kwargs.get("merge_size", self.merge_size)
        size = dict(images_kwargs.get("size", self.size))
        min_pixels = images_kwargs.get("min_pixels", size["shortest_edge"])
        max_pixels = images_kwargs.get("max_pixels", size["longest_edge"])
        if images_kwargs.get("do_resize", self.do_resize):
            height, width = smart_resize(height, width, patch_size * merge_size, min_pixels, max_pixels)
        return (height // patch_size) * (width // patch_size)


def get_video_frame_size(size: SizeDict, num_frames: int, temporal_factor: int, total_max_pixels: int) -> SizeDict:
    """Returns the per-frame pixel budget, shared between preprocessing and patch-count estimation."""
    if min(num_frames, temporal_factor, total_max_pixels) <= 0:
        raise ValueError("Video frame count, temporal factor and total_max_pixels must be positive.")
    if not size.shortest_edge or not size.longest_edge:
        raise ValueError("size must contain shortest_edge and longest_edge pixel budgets.")
    max_pixels = max(size.shortest_edge, min(size.longest_edge, total_max_pixels * temporal_factor // num_frames))
    return SizeDict(shortest_edge=size.shortest_edge, longest_edge=max_pixels)


class MiMoV2VLVideoProcessorInitKwargs(Qwen2VLVideoProcessorInitKwargs, total=False):
    """
    min_pixels (`int`, *optional*, defaults to `8 * 32 * 32`):
        Minimum frame pixel budget. Prefer `size["shortest_edge"]`.
    max_pixels (`int`, *optional*, defaults to `8192 * 32 * 32`):
        Maximum frame pixel budget. Prefer `size["longest_edge"]`.
    patch_size (`int`, *optional*, defaults to 16):
        Spatial patch size of the vision encoder.
    temporal_patch_size (`int`, *optional*, defaults to 2):
        Number of consecutive frames in a temporal patch.
    merge_size (`int`, *optional*, defaults to 2):
        Spatial merge factor along each axis.
    min_frames (`int`, *optional*):
        Minimum frame count when sampling at a target frame rate. No minimum when unset.
    max_frames (`int`, *optional*, defaults to 3600):
        Maximum frame count when sampling at a target frame rate.
    temporal_compression_ratio (`int`, *optional*, defaults to 1):
        Number of temporal grids represented by one timestamp/token group.
    total_max_pixels (`int`, *optional*, defaults to `262144 * 32 * 32`):
        Total pixel budget, measured after temporal compression.
    """

    temporal_compression_ratio: int
    total_max_pixels: int
    cap_pixels_per_frame = AttributeError()
    max_video_tokens = AttributeError()


@auto_docstring(custom_intro="Constructs a video processor for the MiMo V2 VL series.")
class MiMoV2VLVideoProcessor(Qwen2VLVideoProcessor):
    size = {"shortest_edge": 8 * 32 * 32, "longest_edge": 8192 * 32 * 32}
    default_to_square = False
    resample = PILImageResampling.BILINEAR
    patch_size = 16
    do_sample_frames = True
    temporal_compression_ratio = 1
    total_max_pixels = 262144 * 32 * 32
    fps = 1
    num_frames: int | None = None
    min_frames: int | None = None
    max_frames = 3600
    valid_kwargs = MiMoV2VLVideoProcessorInitKwargs
    cap_pixels_per_frame = AttributeError()
    max_video_tokens = AttributeError()

    def _decode_and_sample_videos(
        self,
        videos: VideoInput,
        video_metadata: VideoMetadata | dict,
        do_sample_frames: bool | None = None,
        sample_indices_fn: Callable | None = None,
    ) -> list[torch.Tensor]:
        if video_metadata is not None:
            num_videos = len(make_batched_videos(videos))
            metadata = (
                []
                if isinstance(video_metadata, list) and not video_metadata
                else make_batched_metadata(videos, video_metadata=video_metadata)
            )
            if len(metadata) != num_videos:
                raise ValueError("video_metadata must contain one entry per video.")
        return BaseVideoProcessor._decode_and_sample_videos(
            self, videos, video_metadata, do_sample_frames=do_sample_frames, sample_indices_fn=sample_indices_fn
        )

    def _prepare_input_videos(
        self,
        videos: VideoInput,
        do_convert_rgb: bool | None = None,
        input_data_format: str | ChannelDimension | None = None,
        device: str | None = None,
    ) -> list[torch.Tensor]:
        videos = [
            np.ascontiguousarray(video)
            if isinstance(video, np.ndarray) and any(stride < 0 for stride in video.strides)
            else video
            for video in videos
        ]
        return BaseVideoProcessor._prepare_input_videos(
            self, videos, do_convert_rgb=do_convert_rgb, input_data_format=input_data_format, device=device
        )

    @auto_docstring
    def preprocess(self, videos: VideoInput, **kwargs: Unpack[MiMoV2VLVideoProcessorInitKwargs]) -> BatchFeature:
        if kwargs.get("num_frames") is not None and "fps" not in kwargs:
            kwargs["fps"] = None
        elif kwargs.get("fps") is not None and "num_frames" not in kwargs:
            kwargs["num_frames"] = None
        elif self.num_frames is not None and "fps" not in kwargs:
            kwargs["fps"] = None
        return BaseVideoProcessor.preprocess(self, videos, **kwargs)

    def sample_frames(
        self,
        metadata: VideoMetadata,
        num_frames: int | None = None,
        fps: float | None = None,
        temporal_patch_size: int | None = None,
        min_frames: int | None = None,
        max_frames: int | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Samples frames at rounded, endpoint-exclusive intervals, retaining repeats for short videos."""
        if num_frames is not None and fps is not None:
            raise ValueError("num_frames and fps are mutually exclusive.")
        total_num_frames = metadata.total_num_frames
        if total_num_frames <= 0:
            raise ValueError("Video must contain at least one frame.")
        temporal_patch_size = temporal_patch_size if temporal_patch_size is not None else self.temporal_patch_size
        if temporal_patch_size <= 0:
            raise ValueError("temporal_patch_size must be positive.")
        if num_frames is None and fps is None:
            num_frames = self.num_frames
        if num_frames is not None:
            if not math.isfinite(num_frames) or num_frames <= 0:
                raise ValueError("num_frames must be finite and positive.")
        else:
            fps = self.fps if fps is None else fps
            if any(rate is None or not math.isfinite(rate) or rate <= 0 for rate in (metadata.fps, fps)):
                raise ValueError("A finite positive video_metadata.fps and sampling fps are required.")
            min_frames = self.min_frames if min_frames is None else min_frames
            max_frames = self.max_frames if max_frames is None else max_frames
            num_frames = math.ceil(total_num_frames / metadata.fps * fps)
            if max_frames is not None:
                num_frames = min(num_frames, max_frames)
            if min_frames is not None:
                num_frames = max(num_frames, min_frames)
        num_frames = math.ceil(min(num_frames, total_num_frames) / temporal_patch_size) * temporal_patch_size
        indices = np.linspace(0, total_num_frames, num_frames + 1)[:-1].round().astype(np.int64)
        return np.clip(indices, 0, total_num_frames - 1)

    def resize(
        self,
        videos: torch.Tensor,
        size: SizeDict,
        resample: PILImageResampling,
        factor: int,
        temporal_factor: int | None = None,
        total_max_pixels: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        temporal_factor = (
            self.temporal_patch_size * self.temporal_compression_ratio if temporal_factor is None else temporal_factor
        )
        total_max_pixels = self.total_max_pixels if total_max_pixels is None else total_max_pixels
        batch_size, num_frames = videos.shape[:2]
        video_size = get_video_frame_size(size, num_frames, temporal_factor, total_max_pixels)
        frames = resize_frames(self, videos.flatten(0, 1), video_size, resample, factor)
        return frames.reshape(batch_size, num_frames, *frames.shape[1:])

    def patchify(
        self,
        videos: torch.Tensor,
        patch_size: int,
        merge_size: int,
        temporal_patch_size: int,
        temporal_compression_ratio: int = 1,
    ) -> tuple[torch.Tensor, int, int, int]:
        """
        Patchifies each video into a flat layout of shape (`seq_len`, `patch_dim`), after padding its frames to a
        multiple of `temporal_patch_size * temporal_compression_ratio` by repeating the last frame.
        """
        if min(patch_size, temporal_patch_size, merge_size, temporal_compression_ratio) <= 0:
            raise ValueError("Patch, merge and compression sizes must be positive.")
        if any(length % (patch_size * merge_size) for length in videos.shape[-2:]):
            raise ValueError("Frame dimensions must be divisible by their temporal and spatial patch factors.")
        temporal_factor = temporal_patch_size * temporal_compression_ratio
        if pad_length := -videos.shape[1] % temporal_factor:
            videos = torch.cat((videos, videos[:, -1:].expand(-1, pad_length, -1, -1, -1)), dim=1)

        batch_size, num_frames, channel, resized_height, resized_width = videos.shape
        grid_t = num_frames // temporal_patch_size
        grid_h, grid_w = resized_height // patch_size, resized_width // patch_size
        patches = videos.reshape(
            batch_size,
            grid_t,
            temporal_patch_size,
            channel,
            grid_h // merge_size,
            merge_size,
            patch_size,
            grid_w // merge_size,
            merge_size,
            patch_size,
        )
        patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
        # The patches are contiguous, as the patch embedding of the vision encoder reshapes them with `view`.
        flatten_patches = patches.reshape(
            batch_size, grid_t * grid_h * grid_w, channel * temporal_patch_size * patch_size * patch_size
        ).contiguous()
        return flatten_patches, grid_t, grid_h, grid_w

    def _preprocess(
        self,
        videos: list[torch.Tensor],
        do_resize: bool,
        size: SizeDict,
        resample: PILImageResampling,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float],
        image_std: float | list[float],
        patch_size: int,
        temporal_patch_size: int,
        merge_size: int,
        temporal_compression_ratio: int,
        total_max_pixels: int,
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ) -> BatchFeature:
        temporal_factor = temporal_patch_size * temporal_compression_ratio
        grouped_videos, grouped_indices = group_videos_by_shape(videos)
        processed_videos, processed_grids = {}, {}
        for shape, stacked_videos in grouped_videos.items():
            if stacked_videos.shape[1] == 0:
                raise ValueError("Video must contain at least one frame.")
            if do_resize:
                stacked_videos = self.resize(
                    stacked_videos, size, resample, patch_size * merge_size, temporal_factor, total_max_pixels
                )
            stacked_videos = self.rescale_and_normalize(
                stacked_videos, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            patches, grid_t, grid_h, grid_w = self.patchify(
                stacked_videos, patch_size, merge_size, temporal_patch_size, temporal_compression_ratio
            )
            processed_videos[shape] = patches
            processed_grids[shape] = [[grid_t, grid_h, grid_w]] * len(stacked_videos)
        pixel_values = torch.cat(reorder_videos(processed_videos, grouped_indices), dim=0)
        grids = reorder_videos(processed_grids, grouped_indices)
        return BatchFeature({"pixel_values_videos": pixel_values, "video_grid_thw": grids}, tensor_type=return_tensors)

    def get_num_of_video_patches(
        self, num_frames: int, height: int, width: int, videos_kwargs: dict | None = None
    ) -> int:
        videos_kwargs = videos_kwargs or {}
        patch_size = videos_kwargs.get("patch_size", self.patch_size)
        merge_size = videos_kwargs.get("merge_size", self.merge_size)
        temporal_patch_size = videos_kwargs.get("temporal_patch_size", self.temporal_patch_size)
        compression = videos_kwargs.get("temporal_compression_ratio", self.temporal_compression_ratio)
        total_pixels = videos_kwargs.get("total_max_pixels", self.total_max_pixels)
        if min(num_frames, height, width, patch_size, merge_size, temporal_patch_size, compression, total_pixels) <= 0:
            raise ValueError("Video dimensions, patch sizes, temporal compression and pixel budget must be positive.")
        temporal_factor = temporal_patch_size * compression
        if videos_kwargs.get("do_resize", self.do_resize):
            size = dict(videos_kwargs.get("size", self.size))
            min_pixels = videos_kwargs.get("min_pixels", size["shortest_edge"])
            max_pixels = videos_kwargs.get("max_pixels", size["longest_edge"])
            frame_size = get_video_frame_size(
                SizeDict(shortest_edge=min_pixels, longest_edge=max_pixels), num_frames, temporal_factor, total_pixels
            )
            height, width = smart_resize(
                height, width, patch_size * merge_size, frame_size.shortest_edge, frame_size.longest_edge
            )
        padded_frames = num_frames + -num_frames % temporal_factor
        return (padded_frames // temporal_patch_size) * (height // patch_size) * (width // patch_size)


@lru_cache(maxsize=8)
def _get_mel_filters(n_fft: int, feature_size: int, f_min: float, f_max: float, sampling_rate: int) -> np.ndarray:
    return mel_filter_bank(
        num_frequency_bins=1 + n_fft // 2,
        num_mel_filters=feature_size,
        min_frequency=f_min,
        max_frequency=f_max,
        sampling_rate=sampling_rate,
        norm=None,
        mel_scale="htk",
    )


@lru_cache(maxsize=8)
def _get_spectral_constants(
    n_fft: int,
    win_length: int,
    feature_size: int,
    f_min: float,
    f_max: float,
    sampling_rate: int,
    device: str | None,
):
    """Cache the STFT window and mel filters. A `None` device selects the NumPy backend."""
    filters = _get_mel_filters(n_fft, feature_size, f_min, f_max, sampling_rate)
    if device is None:
        return window_function(win_length, "hann", frame_length=n_fft), filters
    return (
        torch.hann_window(win_length, dtype=torch.float32, device=device),
        torch.as_tensor(filters, dtype=torch.float32, device=device),
    )


class MiMoV2VLFeatureExtractor(SequenceFeatureExtractor):
    r"""
    Constructs an audio feature extractor for the MiMo V2 VL series.

    Extracts the natural logarithm of magnitude mel spectrograms from mono waveforms, or groups pre-tokenized
    audio codec IDs. Use `MiMoV2VLProcessor.from_pretrained("XiaomiMiMo/MiMo-V2.6-Flash-RL")` to load the
    checkpoint's audio settings together with its tokenizer and visual processors.

    This feature extractor inherits from [`SequenceFeatureExtractor`], which provides feature padding and serialization.

    Args:
        feature_size (`int`, *optional*, defaults to 128):
            Number of mel frequency bins.
        sampling_rate (`int`, *optional*, defaults to 24000):
            Waveform sampling rate in hertz.
        n_fft (`int`, *optional*, defaults to 960):
            Size of the Fourier transform.
        hop_length (`int`, *optional*, defaults to 240):
            Number of samples between consecutive STFT frames.
        win_length (`int`, *optional*, defaults to 960):
            Length of the periodic Hann window.
        f_min (`float`, *optional*, defaults to 0.0):
            Lowest mel frequency in hertz.
        f_max (`float`, *optional*):
            Highest mel frequency in hertz. Defaults to half the sampling rate.
        padding_value (`float`, *optional*, defaults to 0.0):
            Value used to pad mel features.
        conv_kernel_size (`int`, *optional*, defaults to 3):
            Kernel size of the input convolutions of the audio tokenizer.
        conv_stride (`int`, *optional*, defaults to 2):
            Stride of the second input convolution of the audio tokenizer.
        downsample_factor (`int`, *optional*, defaults to 2):
            Temporal downsampling factor of the audio tokenizer.
        group_size (`int`, *optional*, defaults to 4):
            Number of consecutive codec frames represented by one audio token.
        num_codebooks (`int`, *optional*, defaults to 20):
            Number of codebooks (residual vector quantization stages of the audio tokenizer) consumed by the model.
        codec_frame_rate (`float`, *optional*, defaults to 25):
            Number of codec frames per second of audio.
        segment_length (`int`, *optional*, defaults to 6000):
            Maximum number of mel frames encoded at once by the audio tokenizer. Longer audios are split into segments,
            whose codec lengths are computed separately.
    """

    model_input_names = ["input_features", "attention_mask", "audio_feature_lens"]

    def __init__(
        self,
        feature_size=128,
        sampling_rate=24000,
        n_fft=960,
        hop_length=240,
        win_length=960,
        f_min=0.0,
        f_max=None,
        padding_value=0.0,
        conv_kernel_size=3,
        conv_stride=2,
        downsample_factor=2,
        group_size=4,
        num_codebooks=20,
        codec_frame_rate=25,
        segment_length=6000,
        **kwargs,
    ):
        kwargs.pop("return_attention_mask", None)
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            return_attention_mask=True,
            **kwargs,
        )
        if (
            min(
                feature_size,
                sampling_rate,
                n_fft,
                hop_length,
                win_length,
                conv_kernel_size,
                conv_stride,
                downsample_factor,
                group_size,
                num_codebooks,
                codec_frame_rate,
                segment_length,
            )
            <= 0
        ):
            raise ValueError("FFT, window, stride, pooling and grouping parameters must be positive.")
        if win_length > n_fft:
            raise ValueError("win_length must not exceed n_fft.")
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.f_min = f_min
        self.f_max = f_max
        self.conv_kernel_size = conv_kernel_size
        self.conv_stride = conv_stride
        self.downsample_factor = downsample_factor
        self.group_size = group_size
        self.num_codebooks = num_codebooks
        self.codec_frame_rate = codec_frame_rate
        self.segment_length = segment_length

    @property
    def audio_tokens_per_second(self) -> float:
        return self.codec_frame_rate / self.group_size

    def _get_audio_token_length(self, mel_length: int) -> int:
        """Compute the token count after audio convolutions, pooling, and codec grouping."""

        def codec_length(length):
            output_length = length + 2 - self.conv_kernel_size + 1  # first convolution: padding 1, stride 1
            output_length = (output_length + 2 - self.conv_kernel_size) // self.conv_stride + 1  # second: padding 1
            return math.ceil(output_length / self.downsample_factor)

        full_segments, remainder = divmod(mel_length, self.segment_length)
        output_length = full_segments * codec_length(self.segment_length)
        if remainder:
            output_length += codec_length(remainder)
        return math.ceil(output_length / self.group_size)

    def _torch_extract_fbank_features(self, waveform: np.ndarray, device: str) -> np.ndarray:
        """Compute a magnitude log-mel spectrogram using PyTorch's STFT."""
        if waveform.strides[0] < 0:
            waveform = np.ascontiguousarray(waveform)
        waveform = torch.as_tensor(waveform, dtype=torch.float32, device=device)
        f_max = self.f_max if self.f_max is not None else self.sampling_rate / 2
        # Include the actual device index in the cache key when callers use device="cuda".
        window, mel_filters = _get_spectral_constants(
            self.n_fft, self.win_length, self.feature_size, self.f_min, f_max, self.sampling_rate, str(waveform.device)
        )
        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True,
        )
        mel_spectrogram = stft.abs().transpose(0, 1) @ mel_filters
        return mel_spectrogram.clamp(min=1e-7).log().cpu().numpy()

    def _np_extract_fbank_features(self, waveform: np.ndarray, device: str) -> np.ndarray:
        """Compute the same magnitude log-mel features using the NumPy STFT backend."""
        if device != "cpu":
            raise ValueError("Feature extraction on an accelerator requires PyTorch.")
        f_max = self.f_max if self.f_max is not None else self.sampling_rate / 2
        window, mel_filters = _get_spectral_constants(
            self.n_fft, self.win_length, self.feature_size, self.f_min, f_max, self.sampling_rate, None
        )
        return spectrogram(
            waveform,
            window,
            frame_length=self.n_fft,
            hop_length=self.hop_length,
            power=1.0,
            mel_filters=mel_filters,
            mel_floor=1e-7,
            log_mel="log",
        ).T

    def __call__(
        self,
        raw_speech: AudioInput,
        sampling_rate: int | None = None,
        padding: bool | str = True,
        max_length: int | None = None,
        truncation: bool = False,
        pad_to_multiple_of: int | None = None,
        return_attention_mask: bool = True,
        return_tensors: str | TensorType | None = None,
        device: str = "cpu",
        is_tokenized: bool = False,
        **kwargs,
    ) -> BatchFeature:
        """
        Prepare waveforms or pre-tokenized codec IDs for the MiMo V2 VL series.

        Args:
            raw_speech (`AudioInput`):
                A mono waveform or batch of waveforms. With `is_tokenized=True`, a sequence of integer codec IDs
                of shape `(time, num_codebooks)`, a list or tuple of such sequences, or a three-dimensional batch.
            sampling_rate (`int`, *optional*):
                Sampling rate of the input waveforms. Must match the extractor's sampling rate.
            padding (`bool` or `str`, *optional*, defaults to `True`):
                Pad to the longest sequence when `True` or `"longest"`, or disable padding with `False` or
                `"do_not_pad"`. Waveforms also support `"max_length"` padding.
            max_length (`int`, *optional*):
                Maximum number of mel frames for waveform padding or truncation.
            truncation (`bool`, *optional*, defaults to `False`):
                Truncate waveform features to `max_length` mel frames.
            pad_to_multiple_of (`int`, *optional*):
                Pad waveform features to a multiple of this number of mel frames.
            return_attention_mask (`bool`, *optional*, defaults to `True`):
                Whether to return a mask distinguishing valid features from padding.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                Return PyTorch tensors with `"pt"`, NumPy arrays with `"np"`, or lists when unset.
            device (`str`, *optional*, defaults to `"cpu"`):
                Device used for the PyTorch STFT. The resulting features are returned on the CPU.
            is_tokenized (`bool`, *optional*, defaults to `False`):
                Process codec IDs instead of waveforms. Incomplete groups repeat the final codec frame.

        Returns:
            [`BatchFeature`]: Waveform `input_features` of shape `(batch, mel_time, mel_bins)`, or `input_audio_codes`
            of shape `(batch, grouped_time, group_size, num_codebooks)`. Also includes `audio_feature_lens` in
            grouped-token units and, when requested, `attention_mask` in feature-frame units.
        """
        if is_tokenized:
            if truncation or max_length is not None or pad_to_multiple_of is not None:
                raise ValueError("Length overrides are only supported for waveform features.")
            output = self._process_audio_codes(raw_speech, padding=padding, return_tensors=return_tensors)
            if not return_attention_mask:
                output.pop("attention_mask")
            return output
        if sampling_rate is not None and sampling_rate != self.sampling_rate:
            raise ValueError(f"Expected sampling_rate={self.sampling_rate}; resample the waveform before calling.")
        waveforms = make_list_of_audio(raw_speech)
        if not waveforms:
            raise ValueError("Audio batch must contain at least one waveform.")
        extract_fbank_features = (
            self._torch_extract_fbank_features if is_torch_available() else self._np_extract_fbank_features
        )
        features = []
        for waveform in waveforms:
            if is_torch_tensor(waveform):
                waveform = waveform.detach().cpu().numpy()
            waveform = np.asarray(waveform, dtype=np.float32)
            if waveform.ndim != 1:
                raise ValueError("Each waveform must be mono and one-dimensional.")
            if not np.isfinite(waveform).all():
                raise ValueError("Waveform samples must be finite.")
            if len(waveform) <= self.n_fft // 2:
                raise ValueError("Waveform is too short for centered STFT reflection padding.")
            features.append(extract_fbank_features(waveform, device))
        output = self.pad(
            BatchFeature({"input_features": features}),
            padding=padding,
            max_length=max_length,
            truncation=truncation,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=True,
            return_tensors=return_tensors,
        )
        mel_lengths = [int(sum(mask) if isinstance(mask, list) else mask.sum()) for mask in output["attention_mask"]]
        output["audio_feature_lens"] = [self._get_audio_token_length(length) for length in mel_lengths]
        if not return_attention_mask:
            output.pop("attention_mask")
        return BatchFeature(dict(output), tensor_type=return_tensors)

    def _process_audio_codes(
        self, audio: AudioInput, padding: bool | str, return_tensors: str | TensorType | None
    ) -> BatchFeature:
        """Group codec frames and pad the batch in grouped-token units."""
        if padding not in (True, False, "longest", "do_not_pad"):
            raise ValueError("Codec inputs support padding=True, False, 'longest', or 'do_not_pad'.")
        padding = padding in (True, "longest")
        if isinstance(audio, np.ndarray) or is_torch_tensor(audio):
            audio = list(audio) if audio.ndim == 3 else [audio]
        elif isinstance(audio, (list, tuple)):
            if audio:
                # A sequence of codec rows is one sample; a sequence of matrices is a batch.
                first = audio[0]
                ndim = first.ndim if isinstance(first, np.ndarray) or is_torch_tensor(first) else np.ndim(first)
                if ndim == 1:
                    audio = [audio]
        else:
            audio = [audio]
        if not audio:
            raise ValueError("Audio batch must contain at least one codec sequence.")
        features, lengths = [], []
        for codes in audio:
            if is_torch_tensor(codes):
                codes = codes.detach().cpu().numpy()
            codes = np.asarray(codes)
            if (
                codes.ndim != 2
                or not len(codes)
                or codes.shape[1] < self.num_codebooks
                or not np.issubdtype(codes.dtype, np.integer)
            ):
                raise ValueError("Audio codes must be nonempty integer (time, num_codebooks) arrays.")
            codes = codes[:, : self.num_codebooks].astype(np.int64, copy=False)
            if pad_length := -len(codes) % self.group_size:
                codes = np.pad(codes, ((0, pad_length), (0, 0)), mode="edge")
            features.append(codes.reshape(-1, self.group_size, self.num_codebooks))
            lengths.append(len(features[-1]))
        max_length = max(lengths)
        if padding:
            features = [
                np.pad(value, ((0, max_length - len(value)), (0, 0), (0, 0))) if len(value) < max_length else value
                for value in features
            ]
        masks = [[1] * length + ([0] * (max_length - length) if padding else []) for length in lengths]
        return BatchFeature(
            {"input_audio_codes": features, "attention_mask": masks, "audio_feature_lens": lengths},
            tensor_type=return_tensors,
        )


class MiMoV2VLAudioProcessorKwargs(AudioKwargs, total=False):
    """
    is_tokenized (`bool`, *optional*, defaults to `False`):
        Interpret audio inputs as integer codec arrays instead of waveforms.
    """

    is_tokenized: bool


class MiMoV2VLVideosKwargs(VideosKwargs, total=False):
    """
    use_audio_in_video (`bool` or `list[bool]`, *optional*, defaults to `False`):
        Whether each video has an audio input. Audio inputs must be supplied in the
        order of standalone audio and audio-enabled video placeholders across the text batch.
    video_audio_interleave_length (`float` or `int`, *optional*):
        Override the processor's audio/video grouping duration in seconds.
    use_per_grid_t_timestamps (`bool`, *optional*):
        Override whether timestamps are repeated for every video/audio grid.
    """

    use_audio_in_video: bool | list[bool]
    video_audio_interleave_length: float | int
    use_per_grid_t_timestamps: bool


class MiMoV2VLProcessorKwargs(ProcessingKwargs, total=False):
    audio_kwargs: MiMoV2VLAudioProcessorKwargs
    videos_kwargs: MiMoV2VLVideosKwargs
    _defaults = {}


@auto_docstring(
    custom_intro="Constructs a multimodal processor for the MiMo V2 VL series.",
    checkpoint="XiaomiMiMo/MiMo-V2.6-Flash-RL",
)
class MiMoV2VLProcessor(ProcessorMixin):
    valid_processor_kwargs = MiMoV2VLProcessorKwargs
    _special_token_names = (
        "image",
        "video",
        "audio",
        "vision_start",
        "vision_end",
        "audio_start",
        "audio_end",
        "video_start",
        "video_end",
    )

    def __init__(
        self,
        image_processor=None,
        video_processor=None,
        feature_extractor=None,
        tokenizer=None,
        chat_template=None,
        image_token="<|image_pad|>",
        video_token="<|video_pad|>",
        audio_token="<|audio_pad|>",
        vision_start_token="<|vision_start|>",
        vision_end_token="<|vision_end|>",
        audio_start_token="<|mimo_audio_start|>",
        audio_end_token="<|mimo_audio_end|>",
        video_start_token="<|mimo_video_start|>",
        video_end_token="<|mimo_video_end|>",
        video_audio_interleave_length=0,
        use_per_grid_t_timestamps=False,
    ):
        r"""
        image_processor ([`MiMoV2VLImageProcessor`], *optional*):
            Image processor. Defaults to a new `MiMoV2VLImageProcessor` when unset.
        video_processor ([`MiMoV2VLVideoProcessor`], *optional*):
            Video processor. Defaults to a new `MiMoV2VLVideoProcessor` when unset.
        feature_extractor ([`MiMoV2VLFeatureExtractor`], *optional*):
            Audio feature extractor. Defaults to a new `MiMoV2VLFeatureExtractor` when unset.
        chat_template (`str`, *optional*):
            Jinja template for chat messages. Defaults to the tokenizer's chat template when available.
        image_token (`str`, *optional*, defaults to `"<|image_pad|>"`):
            Image placeholder token.
        video_token (`str`, *optional*, defaults to `"<|video_pad|>"`):
            Video placeholder token.
        audio_token (`str`, *optional*, defaults to `"<|audio_pad|>"`):
            Audio placeholder token.
        vision_start_token (`str`, *optional*, defaults to `"<|vision_start|>"`):
            Start of an image or video-frame feature span.
        vision_end_token (`str`, *optional*, defaults to `"<|vision_end|>"`):
            End of an image or video-frame feature span.
        audio_start_token (`str`, *optional*, defaults to `"<|mimo_audio_start|>"`):
            Start of an audio feature span.
        audio_end_token (`str`, *optional*, defaults to `"<|mimo_audio_end|>"`):
            End of an audio feature span.
        video_start_token (`str`, *optional*, defaults to `"<|mimo_video_start|>"`):
            Start of the complete timestamped video sequence.
        video_end_token (`str`, *optional*, defaults to `"<|mimo_video_end|>"`):
            End of the complete timestamped video sequence.
        video_audio_interleave_length (`float`, *optional*, defaults to 0):
            Group one video grid followed by audio when zero, all video followed by audio
            when -1, or group grids into intervals of this many seconds when positive.
        use_per_grid_t_timestamps (`bool`, *optional*, defaults to `False`):
            Emit a timestamp for every video and audio grid rather than once per group.
        """
        image_processor = image_processor if image_processor is not None else MiMoV2VLImageProcessor()
        video_processor = video_processor if video_processor is not None else MiMoV2VLVideoProcessor()
        feature_extractor = feature_extractor if feature_extractor is not None else MiMoV2VLFeatureExtractor()
        if chat_template is None:
            chat_template = getattr(tokenizer, "chat_template", None)
        super().__init__(image_processor, video_processor, feature_extractor, tokenizer, chat_template=chat_template)
        self.image_token = image_token
        self.video_token = video_token
        self.audio_token = audio_token
        self.vision_start_token = vision_start_token
        self.vision_end_token = vision_end_token
        self.audio_start_token = audio_start_token
        self.audio_end_token = audio_end_token
        self.video_start_token = video_start_token
        self.video_end_token = video_end_token
        for name in self._special_token_names:
            token = getattr(self, f"{name}_token")
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            if token_id == self.tokenizer.unk_token_id:
                token_id = None
            setattr(self, f"{name}_token_id", token_id)
        self._validate_interleave_length(video_audio_interleave_length)
        self.video_audio_interleave_length = video_audio_interleave_length
        self.use_per_grid_t_timestamps = use_per_grid_t_timestamps

    def apply_chat_template(self, conversation, chat_template=None, **kwargs):
        # The chat template would place the loaded tracks after the videos, instead of interleaving them.
        if kwargs.get("load_audio_from_video"):
            raise ValueError(
                "`load_audio_from_video` is not supported: pass the audio track of each video through `audio` in a "
                "processor call with `use_audio_in_video=True`, to interleave it with the frames."
            )
        return super().apply_chat_template(conversation, chat_template=chat_template, **kwargs)

    def _validate_modality_tokens(self, modality: str) -> None:
        names = {
            "image": ("image", "vision_start", "vision_end"),
            "video": ("video", "vision_start", "vision_end", "video_start", "video_end"),
            "audio": ("audio", "audio_start", "audio_end"),
        }[modality]
        for name in names:
            token = getattr(self, f"{name}_token")
            token_id = getattr(self, f"{name}_token_id")
            if token_id is None or self.tokenizer.encode(token, add_special_tokens=False) != [token_id]:
                raise ValueError(f"Tokenizer is missing the MiMo V2 series {modality} special token {token!r}.")

    @staticmethod
    def _validate_interleave_length(length: float) -> None:
        if not math.isfinite(length) or (length < 0 and length != -1):
            raise ValueError("video_audio_interleave_length must be -1 or a finite nonnegative duration.")

    def _get_multimodal_token_pattern(self) -> re.Pattern:
        """Match bare placeholders and placeholders already framed by the checkpoint's chat template."""
        patterns = []
        for modality in ("image", "video", "audio"):
            prefix = "audio" if modality == "audio" else "vision"
            token = re.escape(getattr(self, f"{modality}_token"))
            start, end = (
                re.escape(getattr(self, f"{prefix}_start_token")),
                re.escape(getattr(self, f"{prefix}_end_token")),
            )
            patterns.append(f"(?P<{modality}>{start}(?:{token})+{end}|{token})")
        return re.compile("|".join(patterns))

    @auto_docstring
    def __call__(
        self,
        text: TextInput | list[TextInput] | None = None,
        images: ImageInput | None = None,
        videos: VideoInput | None = None,
        audio: AudioInput | None = None,
        **kwargs: Unpack[MiMoV2VLProcessorKwargs],
    ) -> BatchFeature:
        r"""
        videos (`VideoInput`, *optional*):
            Videos to preprocess. Decoded frame arrays require `video_metadata` containing the original frame
            rate and frame indices. Set `do_sample_frames=False` for already sampled frames.
        audio (`AudioInput`, *optional*):
            Audio waveforms, or codec IDs when `is_tokenized=True`. Pass video audio tracks explicitly with
            `use_audio_in_video=True` or one boolean per video, in placeholder order across the text batch.

        Returns:
            [`BatchFeature`]: Tokenized text and the supplied modalities' features. Waveforms produce `input_features`
            of shape `(batch, mel_time, mel_bins)`; codec inputs produce `input_audio_codes` of shape
            `(batch, grouped_time, group_size, num_codebooks)`. `audio_feature_lens` counts audio tokens and
            `feature_attention_mask` marks valid feature frames. `audio_segment_offsets` contains
            `(audio_index, start, length)` rows in grouped-token units, ordered as the audio spans in the prompt.
        """
        output_kwargs = self._merge_kwargs(
            self.valid_processor_kwargs, tokenizer_init_kwargs=self.tokenizer.init_kwargs, **kwargs
        )
        video_kwargs = output_kwargs["videos_kwargs"]
        audio_kwargs = output_kwargs["audio_kwargs"]
        text_kwargs = output_kwargs["text_kwargs"]
        use_audio_in_video = video_kwargs.pop("use_audio_in_video", False)
        interleave_length = video_kwargs.pop("video_audio_interleave_length", self.video_audio_interleave_length)
        use_per_grid_t_timestamps = video_kwargs.pop("use_per_grid_t_timestamps", self.use_per_grid_t_timestamps)
        # Like the original processor: timestamps have milliseconds when sampling more than 2 frames per second, and
        # sampled videos end at their duration.
        sampling_fps = video_kwargs.get("fps")
        use_ms_timestamps = sampling_fps is not None and sampling_fps > 2
        do_sample_frames = video_kwargs.get("do_sample_frames", self.video_processor.do_sample_frames)
        self._validate_interleave_length(interleave_length)
        is_tokenized_audio = audio_kwargs.get("is_tokenized", False)
        images, text, videos, fetched_audio = self.prepare_inputs_layout(
            images=images,
            text=text,
            videos=videos,
            audio=None if is_tokenized_audio else audio,
            sampling_rate=audio_kwargs.get("sampling_rate", self.feature_extractor.sampling_rate),
        )
        audio = audio if is_tokenized_audio else fetched_audio
        self.validate_inputs(images=images, text=text, videos=videos, audio=audio, **kwargs)
        for modality, inputs in (("image", images), ("video", videos), ("audio", audio)):
            if inputs is not None:
                self._validate_modality_tokens(modality)
        image_inputs = self.image_processor(images, **output_kwargs["images_kwargs"]) if images is not None else {}
        video_kwargs["return_metadata"] = True
        video_inputs = self.video_processor(videos, **video_kwargs) if videos is not None else {}
        audio_inputs = self.feature_extractor(audio, **audio_kwargs) if audio is not None else {}
        if "attention_mask" in audio_inputs:
            audio_inputs["feature_attention_mask"] = audio_inputs.pop("attention_mask")

        num_videos = len(video_inputs.get("video_grid_thw", []))
        if len(video_inputs.get("video_metadata", [])) != num_videos:
            raise ValueError("video_metadata must contain one entry per video.")
        use_audio_in_video = (
            [use_audio_in_video] * num_videos if isinstance(use_audio_in_video, bool) else list(use_audio_in_video)
        )
        if len(use_audio_in_video) != num_videos or not all(isinstance(flag, bool) for flag in use_audio_in_video):
            raise ValueError("use_audio_in_video must contain one flag per video.")
        if text is None and any(use_audio_in_video):
            raise ValueError("Text is required to align audio inputs with video placeholders.")
        text_inputs = {}
        audio_segment_offsets = []
        if text is not None:
            return_replacement_offsets = text_kwargs.pop("return_text_replacement_offsets", False)
            text, audio_segment_offsets, replacement_offsets = self._expand_multimodal_tokens(
                text,
                image_inputs=image_inputs,
                video_inputs=video_inputs,
                audio_inputs=audio_inputs,
                images_kwargs=output_kwargs["images_kwargs"],
                videos_kwargs=video_kwargs,
                use_audio_in_video=use_audio_in_video,
                interleave_length=interleave_length,
                use_per_grid_t_timestamps=use_per_grid_t_timestamps,
                use_ms_timestamps=use_ms_timestamps,
                do_sample_frames=do_sample_frames,
                return_replacement_offsets=return_replacement_offsets,
            )
            return_token_types = text_kwargs.pop("return_mm_token_type_ids", False)
            text_inputs = self.tokenizer(text, **text_kwargs)
            self._check_special_mm_tokens(text, text_inputs, modalities=["image", "video", "audio"])
            if return_token_types:
                text_inputs["mm_token_type_ids"] = self.create_mm_token_type_ids(text_inputs["input_ids"])
            if return_replacement_offsets:
                text_inputs["text_replacement_offsets"] = replacement_offsets
        data = {**text_inputs, **image_inputs, **video_inputs, **audio_inputs}
        if audio_segment_offsets:
            data["audio_segment_offsets"] = audio_segment_offsets
        if not kwargs.get("return_metadata", kwargs.get("videos_kwargs", {}).get("return_metadata", False)):
            data.pop("video_metadata", None)
        return BatchFeature(
            data, tensor_type=text_kwargs.get("return_tensors"), skip_tensor_conversion=self.skip_tensor_conversion
        )

    def _expand_multimodal_tokens(
        self,
        text: list[str],
        image_inputs: BatchFeature | dict,
        video_inputs: BatchFeature | dict,
        audio_inputs: BatchFeature | dict,
        images_kwargs: dict,
        videos_kwargs: dict,
        use_audio_in_video: list[bool],
        interleave_length: float,
        use_per_grid_t_timestamps: bool,
        use_ms_timestamps: bool = False,
        do_sample_frames: bool = True,
        return_replacement_offsets: bool = False,
    ) -> tuple[list[str], list[tuple[int, int, int]], list[list[dict]]]:
        """Expand placeholders in prompt order and collect the corresponding audio segments."""
        image_grid_thw = image_inputs.get("image_grid_thw", [])
        video_grid_thw = video_inputs.get("video_grid_thw", [])
        audio_lengths = audio_inputs.get("audio_feature_lens", [])
        image_merge_size = images_kwargs.get("merge_size", self.image_processor.merge_size)
        video_merge_size = videos_kwargs.get("merge_size", self.video_processor.merge_size)
        temporal_patch_size = videos_kwargs.get("temporal_patch_size", self.video_processor.temporal_patch_size)
        temporal_compression_ratio = videos_kwargs.get(
            "temporal_compression_ratio", self.video_processor.temporal_compression_ratio
        )
        counts = {"image": 0, "video": 0, "audio": 0}
        expected_counts = {"image": len(image_grid_thw), "video": len(video_grid_thw), "audio": len(audio_lengths)}
        audio_segment_offsets = []

        def replace_placeholder(match):
            modality = match.lastgroup
            index = counts[modality]
            if index >= expected_counts[modality]:
                raise ValueError(f"More {modality} placeholders than {modality} inputs.")
            counts[modality] += 1

            if modality == "image":
                num_image_tokens = math.prod(int(size) for size in image_grid_thw[index]) // image_merge_size**2
                return self.vision_start_token + self.image_token * num_image_tokens + self.vision_end_token

            if modality == "video":
                grid_t, grid_h, grid_w = (int(size) for size in video_grid_thw[index])
                timestamps, video_end_time = self._get_video_timestamps(
                    video_inputs["video_metadata"][index],
                    grid_t=grid_t,
                    temporal_patch_size=temporal_patch_size,
                    temporal_compression_ratio=temporal_compression_ratio,
                    use_video_duration=do_sample_frames,
                )
                num_video_tokens = grid_h * grid_w // video_merge_size**2
                if not use_audio_in_video[index]:
                    frames = [
                        self._format_timestamp(timestamp, use_ms_timestamps)
                        + self.vision_start_token
                        + self.video_token * num_video_tokens
                        + self.vision_end_token
                        for timestamp in timestamps
                    ]
                    return self.video_start_token + "".join(frames) + self.video_end_token

                audio_index = counts["audio"]
                if audio_index >= len(audio_lengths):
                    raise ValueError("Missing audio for an audio-enabled video.")
                counts["audio"] += 1
                video_text, video_audio_segments = self._build_video_audio_text(
                    timestamps,
                    video_end_time=video_end_time,
                    num_video_tokens=num_video_tokens,
                    audio_length=int(audio_lengths[audio_index]),
                    audio_index=audio_index,
                    interleave_length=interleave_length,
                    use_per_grid_t_timestamps=use_per_grid_t_timestamps,
                    use_ms_timestamps=use_ms_timestamps,
                )
                audio_segment_offsets.extend(video_audio_segments)
                return video_text

            audio_length = int(audio_lengths[index])
            audio_segment_offsets.append((index, 0, audio_length))
            return self.audio_start_token + self.audio_token * audio_length + self.audio_end_token

        pattern = self._get_multimodal_token_pattern()
        expanded_text, batch_replacement_offsets = [], []
        for sample in text:
            pieces, replacement_offsets, last, shift = [], [], 0, 0
            for match in pattern.finditer(sample):
                start, end = match.span()
                replacement = replace_placeholder(match)
                pieces.append(sample[last:start] + replacement)
                if return_replacement_offsets:
                    replacement_offsets.append(
                        {
                            "type": match.lastgroup,
                            "span": (start, end),
                            "new_span": (start + shift, start + shift + len(replacement)),
                            "text": match.group(),
                            "replacement": replacement,
                        }
                    )
                shift += len(replacement) - (end - start)
                last = end
            expanded_text.append("".join(pieces) + sample[last:])
            batch_replacement_offsets.append(replacement_offsets)
        if counts != expected_counts:
            raise ValueError("Every image, video and audio input must have a matching placeholder.")
        return expanded_text, audio_segment_offsets, batch_replacement_offsets

    @staticmethod
    def _format_timestamp(timestamp: float, with_ms: bool = False) -> str:
        minutes, seconds = int(timestamp // 60), int(timestamp % 60)
        if not with_ms:
            return f"{minutes:02d}:{seconds:02d}"
        milliseconds = int(round((timestamp - int(timestamp)) * 1000))
        return f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

    @staticmethod
    def _get_video_timestamps(
        metadata: VideoMetadata,
        grid_t: int,
        temporal_patch_size: int,
        temporal_compression_ratio: int,
        use_video_duration: bool = False,
    ) -> tuple[np.ndarray, float]:
        """
        Return timestamps for compressed temporal grids and the exclusive end of the video: its duration when the
        frames were sampled from the decoded video, or one frame interval after the last frame otherwise.
        """
        if metadata.fps is None or not math.isfinite(metadata.fps) or metadata.fps <= 0:
            raise ValueError("Video timestamps require video_metadata with a finite positive fps.")
        timestamps = np.asarray(metadata.timestamps, dtype=np.float32)
        if (
            not len(timestamps)
            or not np.isfinite(timestamps).all()
            or np.any(timestamps < 0)
            or np.any(np.diff(timestamps) < 0)
        ):
            raise ValueError("Video timestamps must be finite and nondecreasing.")
        if grid_t % temporal_compression_ratio:
            raise ValueError("Video grid length must be divisible by temporal_compression_ratio.")
        temporal_factor = temporal_patch_size * temporal_compression_ratio
        num_timestamp_groups = grid_t // temporal_compression_ratio
        if math.ceil(len(timestamps) / temporal_factor) != num_timestamp_groups:
            raise ValueError("Video metadata frame indices do not match the processed frames.")
        indices = np.minimum(np.arange(num_timestamp_groups) * temporal_factor, len(timestamps) - 1)
        if use_video_duration and metadata.total_num_frames:
            return timestamps[indices], metadata.total_num_frames / metadata.fps
        interval = timestamps[1] - timestamps[0] if len(timestamps) > 1 else 1.0
        if interval <= 0:
            interval = 1.0
        return timestamps[indices], timestamps[-1] + interval

    def _build_video_audio_text(
        self,
        timestamps: np.ndarray,
        video_end_time: float,
        num_video_tokens: int,
        audio_length: int,
        audio_index: int,
        interleave_length: float,
        use_per_grid_t_timestamps: bool,
        use_ms_timestamps: bool = False,
    ) -> tuple[str, list[tuple[int, int, int]]]:
        """Build an interleaved video/audio prompt and its grouped-codec offsets."""
        audio_tokens_per_second = self.feature_extractor.audio_tokens_per_second
        end_timestamps = list(timestamps[1:]) + [video_end_time]
        units = []
        audio_segment_offsets = []
        for timestamp, next_timestamp in zip(timestamps, end_timestamps):
            # The audio track may be slightly shorter than the frames: the last audio units are then empty.
            start = min(int(timestamp * audio_tokens_per_second), audio_length)
            length = min(int(next_timestamp * audio_tokens_per_second), audio_length) - start
            units.append((timestamp, length))
            audio_segment_offsets.append((audio_index, start, length))
        if interleave_length == -1:
            groups = [units]
        elif interleave_length == 0:
            groups = [[unit] for unit in units]
        else:
            groups = [
                list(group)
                for _, group in groupby(units, key=lambda unit: math.floor(float(unit[0]) / interleave_length))
            ]
        output = [self.video_start_token]
        for group in groups:
            if not use_per_grid_t_timestamps:
                output.append(self._format_timestamp(group[0][0], use_ms_timestamps))
            video_tokens = []
            audio_tokens = []
            for timestamp, audio_count in group:
                prefix = self._format_timestamp(timestamp, use_ms_timestamps) if use_per_grid_t_timestamps else ""
                video_tokens.append(
                    prefix + self.vision_start_token + self.video_token * num_video_tokens + self.vision_end_token
                )
                audio_tokens.append(prefix + self.audio_token * audio_count)
            output.extend(("".join(video_tokens), self.audio_start_token, "".join(audio_tokens), self.audio_end_token))
        output.append(self.video_end_token)
        return "".join(output), audio_segment_offsets

    @property
    def model_input_names(self):
        return list(
            dict.fromkeys(
                self.tokenizer.model_input_names
                + self.image_processor.model_input_names
                + self.video_processor.model_input_names
                + [
                    "input_features",
                    "input_audio_codes",
                    "feature_attention_mask",
                    "audio_feature_lens",
                    "audio_segment_offsets",
                    "mm_token_type_ids",
                ]
            )
        )


__all__ = [
    "MiMoV2VLConfig",
    "MiMoV2VLTextConfig",
    "MiMoV2VLVisionConfig",
    "MiMoV2VLAudioConfig",
    "MiMoV2VLAudioTokenizerConfig",
    "MiMoV2VLPreTrainedModel",
    "MiMoV2VLTextModel",
    "MiMoV2VLModel",
    "MiMoV2VLForConditionalGeneration",
    "MiMoV2VLImageProcessor",
    "MiMoV2VLVideoProcessor",
    "MiMoV2VLFeatureExtractor",
    "MiMoV2VLProcessor",
]
