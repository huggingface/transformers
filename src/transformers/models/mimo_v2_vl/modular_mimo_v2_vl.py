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
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from huggingface_hub.dataclasses import strict
from torch import nn
from torch.nn import functional as F

from ... import initialization as init
from ...activations import ACT2FN
from ...cache_utils import Cache
from ...configuration_utils import PreTrainedConfig, SubConfigSpec
from ...generation import GenerationMixin
from ...image_processing_backends import TorchvisionBackend
from ...image_processing_utils import BatchFeature
from ...image_utils import ChannelDimension, ImageInput, PILImageResampling, SizeDict
from ...masking_utils import create_causal_mask
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    BaseModelOutputWithPooling,
    CausalLMOutputWithPast,
    ModelOutput,
)
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, slice_logits_to_keep
from ...processing_utils import Unpack
from ...utils import TensorType, TransformersKwargs, auto_docstring, can_return_tuple, logging, torch_compilable_check
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
from ..glm4_moe.modeling_glm4_moe import apply_rotary_pos_emb
from ..llama.modeling_llama import eager_attention_forward
from ..mimo_v2_flash.configuration_mimo_v2_flash import MiMoV2FlashConfig
from ..mimo_v2_flash.modeling_mimo_v2_flash import (
    MiMoV2FlashAttention,
    MiMoV2FlashDecoderLayer,
    MiMoV2FlashModel,
    MiMoV2FlashPreTrainedModel,
    MiMoV2FlashRMSNorm,
    MiMoV2FlashRotaryEmbedding,
)
from ..qwen2.modeling_qwen2 import Qwen2Attention, Qwen2DecoderLayer, Qwen2MLP, Qwen2RotaryEmbedding
from ..qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLMLP,
    Qwen2_5_VLVisionBlock,
    Qwen2_5_VLVisionRotaryEmbedding,
    apply_rotary_pos_emb_vision,
)
from ..qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor, Qwen2VLImageProcessorKwargs
from ..qwen2_vl.video_processing_qwen2_vl import Qwen2VLVideoProcessor, Qwen2VLVideoProcessorInitKwargs
from ..qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionPatchEmbed, Qwen3VLVisionPatchMerger
from ..whisper.modeling_whisper import WhisperAttention


logger = logging.get_logger(__name__)


# --------------------------------------------------------------------------------------------------------------------
# Configurations
# --------------------------------------------------------------------------------------------------------------------


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
class MiMoV2VLVisionConfig(PreTrainedConfig):
    r"""
    num_key_value_heads (`int`, *optional*, defaults to 8):
        Number of key/value heads of the grouped-query attention.
    head_dim (`int`, *optional*, defaults to 64):
        Dimension of the attention heads, independent of `hidden_size // num_heads`.
    out_hidden_size (`int`, *optional*, defaults to 4096):
        Output dimension of the patch merger; must match the language model's hidden size.
    merger_bias (`bool`, *optional*, defaults to `False`):
        Whether the linear layers of the patch merger have biases.
    layer_types (`list[str]`, *optional*):
        Attention type of each block: `"full_attention"` (every patch of the image or frame) or `"sliding_attention"`
        (the `sliding_window` patches on each side, with learned attention sinks if `use_attention_sinks`).
    sliding_window (`int`, *optional*, defaults to 64):
        Number of patches visible on each side of a query in `"sliding_attention"` blocks.
    use_attention_sinks (`bool`, *optional*, defaults to `True`):
        Whether `"sliding_attention"` blocks have learned attention sinks.
    patch_orders (`list[str]`, *optional*):
        Order in which each block traverses the patches of every image or frame: `"row_major"` (as produced by the
        processor) or `"column_major"`. This matters for the windows of `"sliding_attention"` blocks.
    """

    model_type = "mimo_v2_vl_vision"
    base_config_key = "vision_config"
    default_rope_type = "axial"
    attribute_map = {"num_attention_heads": "num_heads", "num_hidden_layers": "depth"}

    depth: int = 28
    hidden_size: int = 1280
    intermediate_size: int = 4608
    hidden_act: str = "silu"
    num_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 64
    in_channels: int = 3
    patch_size: int = 16
    temporal_patch_size: int = 2
    spatial_merge_size: int = 2
    out_hidden_size: int = 4096
    rms_norm_eps: float = 1e-6
    merger_bias: bool = False
    initializer_range: float = 0.02
    layer_types: list[str] | None = None
    sliding_window: int = 64
    use_attention_sinks: bool = True
    patch_orders: list[str] | None = None
    rope_parameters: dict | None = None

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
        super().__post_init__(**kwargs)

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
    is_causal (`bool`, *optional*, defaults to `False`):
        Whether the codec frames of a group attend causally instead of bidirectionally.
    use_final_norm (`bool`, *optional*, defaults to `True`):
        Whether to apply RMS normalization after the last layer.
    out_hidden_size (`int`, *optional*, defaults to 4096):
        Output dimension of the audio projection; must match the language model's hidden size.
    projection_layers (`int`, *optional*, defaults to 2):
        Number of linear layers of the audio projection, either one or two.
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
    is_causal: bool = False
    use_final_norm: bool = True
    rms_norm_eps: float = 1e-6
    out_hidden_size: int = 4096
    projection_layers: int = 2
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
        if self.projection_layers not in (1, 2):
            raise ValueError(f"`projection_layers` must be 1 or 2, got {self.projection_layers}.")
        for name in ("codebook_vocab_sizes", "codebook_pad_token_ids"):
            if len(getattr(self, name)) != self.num_codebooks:
                raise ValueError(f"`{name}` must contain one value per codebook ({self.num_codebooks}).")


@auto_docstring(checkpoint="XiaomiMiMo/MiMo-V2.6-Flash-RL")
@strict
class MiMoV2VLAudioTokenizerConfig(PreTrainedConfig):
    r"""
    num_mel_bins (`int`, *optional*, defaults to 128):
        Number of mel bins of the input features.
    conv_kernel_size (`int`, *optional*, defaults to 3):
        Kernel size of the two input convolutions.
    conv_stride (`int`, *optional*, defaults to 2):
        Stride of the second input convolution.
    is_causal (`bool`, *optional*, defaults to `True`):
        Whether the encoder layers attend causally.
    layer_types (`list[str]`, *optional*):
        Attention type of each layer: `"sliding_attention"` (the `sliding_window` previous frames) or
        `"full_attention"`. Defaults to alternating sliding and full attention layers.
    sliding_window (`int`, *optional*, defaults to 128):
        Number of previous frames visible to each frame in `"sliding_attention"` layers.
    skip_layer_idx (`int`, *optional*, defaults to 2):
        Index of the layer whose output is added to the output of the last layer. `None` disables the skip connection.
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
    num_mel_bins: int = 128
    conv_kernel_size: int = 3
    conv_stride: int = 2
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    intermediate_size: int = 4096
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-5
    is_causal: bool = True
    layer_types: list[str] | None = None
    sliding_window: int = 128
    skip_layer_idx: int | None = 2
    downsample_factor: int = 2
    codebook_sizes: list[int] | None = None
    segment_length: int = 6000
    rope_parameters: dict | None = None
    initializer_range: float = 0.02

    def __post_init__(self, **kwargs):
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention" if i % 2 == 0 else "full_attention" for i in range(self.num_hidden_layers)
            ]
        if self.codebook_sizes is None:
            self.codebook_sizes = [1024, 1024, 256] + [128] * 17
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("`hidden_size` must be divisible by `num_attention_heads`.")


@auto_docstring(checkpoint="XiaomiMiMo/MiMo-V2.6-Flash-RL")
@strict
class MiMoV2VLConfig(PreTrainedConfig):
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
    keys_to_ignore_at_inference = ["past_key_values"]

    text_config: dict | PreTrainedConfig | None = None
    vision_config: dict | PreTrainedConfig | None = None
    audio_config: dict | PreTrainedConfig | None = None
    audio_tokenizer_config: dict | PreTrainedConfig | None = None
    image_token_id: int = 151655
    video_token_id: int = 151656
    audio_token_id: int = 151669
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    audio_start_token_id: int = 151673
    audio_end_token_id: int = 151674
    tie_word_embeddings: bool = False
    initializer_range: float = 0.02
    pad_token_id: int | None = None
    eos_token_id: int | list[int] | None = None

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
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


# --------------------------------------------------------------------------------------------------------------------
# Shared building blocks
# --------------------------------------------------------------------------------------------------------------------


class MiMoV2VLRMSNorm(MiMoV2FlashRMSNorm):
    pass


def apply_rotary_pos_emb_fp32(query, key, cos, sin):
    """Rotates the rotary part of queries and keys in float32 and leaves their remaining channels untouched."""
    rotary_dim = cos.shape[-1]
    query_rotary, key_rotary = apply_rotary_pos_emb(
        query[..., :rotary_dim].float(), key[..., :rotary_dim].float(), cos, sin
    )
    query = torch.cat((query_rotary.to(query.dtype), query[..., rotary_dim:]), dim=-1)
    key = torch.cat((key_rotary.to(key.dtype), key[..., rotary_dim:]), dim=-1)
    return query, key


def add_attention_sinks(
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    sinks: torch.Tensor,
    query: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Adds learned attention sinks, i.e. one extra logit per head in the softmax denominator, as an additional key whose
    value is zero and whose logit is set by the attention mask. This works with any attention implementation.
    """
    batch_size, num_heads, query_length, _ = query.shape
    key_length = key.shape[-2]
    if attention_mask is None:
        attention_mask = query.new_zeros(batch_size, 1, query_length, key_length)
    attention_mask = attention_mask.expand(batch_size, num_heads, query_length, key_length)
    sink_logits = sinks.to(query.dtype).view(1, num_heads, 1, 1).expand(batch_size, -1, query_length, -1)
    attention_mask = torch.cat((attention_mask, sink_logits), dim=-1)
    key = F.pad(key, (0, 0, 0, 1))
    value = F.pad(value, (0, 0, 0, 1))
    return key, value, attention_mask


def sink_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    window_size: tuple[int, int] | None = None,
    query_chunk_size: int = 512,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """
    Attention with the learned sinks of `module` (if any), optionally restricted to a local window, computed with the
    attention implementation of `module.config`.

    `window_size=(left, right)` limits each query to the keys at most `left` positions before and `right` positions
    after it (-1 means unbounded). Queries are aligned with the end of the keys, as in both prefill and cached decoding.
    Long or windowed inputs are processed in chunks of `query_chunk_size` queries, each only reading the keys inside its
    window, so that the memory of local attention stays linear in the sequence length.
    """
    attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
        module.config._attn_implementation, eager_attention_forward
    )
    sinks = getattr(module, "sinks", None)
    query_length, key_length = query.shape[-2], key.shape[-2]
    offset = key_length - query_length
    if window_size is None:
        query_chunk_size = max(query_chunk_size, query_length)

    attn_outputs = []
    for start in range(0, query_length, query_chunk_size):
        end = min(start + query_chunk_size, query_length)
        key_start, key_end = 0, key_length
        if window_size is not None:
            left, right = window_size
            if left >= 0:
                key_start = max(0, offset + start - left)
            if right >= 0:
                key_end = min(key_length, offset + end + right)

        mask = attention_mask
        if mask is not None:
            if mask.shape[-2] != 1:
                mask = mask[..., start:end, :]
            mask = mask[..., key_start:key_end]
        if window_size is not None:
            query_positions = torch.arange(start, end, device=query.device) + offset
            key_positions = torch.arange(key_start, key_end, device=query.device)
            distance = key_positions[None] - query_positions[:, None]
            allowed = torch.ones_like(distance, dtype=torch.bool)
            if left >= 0:
                allowed &= distance >= -left
            if right >= 0:
                allowed &= distance <= right
            if mask is None:
                mask = allowed
            elif mask.dtype == torch.bool:
                mask = mask & allowed
            else:
                mask = mask.masked_fill(~allowed, -torch.inf)
        if mask is not None and mask.dtype == torch.bool:
            mask = torch.zeros_like(mask, dtype=query.dtype).masked_fill(~mask, -torch.inf)

        query_chunk = query[..., start:end, :]
        key_chunk = key[..., key_start:key_end, :]
        value_chunk = value[..., key_start:key_end, :]
        if sinks is not None:
            key_chunk, value_chunk, mask = add_attention_sinks(key_chunk, value_chunk, mask, sinks, query_chunk)
        attn_output, _ = attention_interface(
            module,
            query_chunk,
            key_chunk,
            value_chunk,
            mask,
            scaling=scaling,
            dropout=dropout,
            is_causal=False,
            **kwargs,
        )
        attn_outputs.append(attn_output)
    return torch.cat(attn_outputs, dim=1), None


# --------------------------------------------------------------------------------------------------------------------
# Language model
# --------------------------------------------------------------------------------------------------------------------


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

        # The mask builders skip the causal mask when SDPA could use `is_causal`, which cannot be combined with the
        # additional key of the attention sinks: build it explicitly.
        query_length, key_length = query_states.shape[-2], key_states.shape[-2]
        if attention_mask is None and query_length > 1:
            query_positions = torch.arange(query_length, device=query_states.device) + key_length - query_length
            key_positions = torch.arange(key_length, device=query_states.device)
            attention_mask = query_states.new_zeros(query_length, key_length).masked_fill(
                key_positions[None, :] > query_positions[:, None], -torch.inf
            )[None, None]

        # The sliding-window mask already hides the keys outside the window. Passing `window_size` as well lets each
        # chunk of queries only read the keys inside its window, which keeps long-context prefill memory linear.
        attn_output, attn_weights = sink_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.attention_dropout,
            window_size=(self.sliding_window - 1, 0) if self.sliding_window is not None else None,
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
    _supports_sdpa = True
    _supports_flash_attn = False
    _supports_flex_attn = False
    _supports_attention_backend = False
    _can_compile_fullgraph = False
    _compatible_flash_implementations = AttributeError()
    _no_split_modules = [
        "MiMoV2VLDecoderLayer",
        "MiMoV2VLVisionBlock",
        "MiMoV2VLAudioEncoderLayer",
        "MiMoV2VLAudioTokenizerLayer",
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


# --------------------------------------------------------------------------------------------------------------------
# Vision encoder
# --------------------------------------------------------------------------------------------------------------------


class MiMoV2VLVisionMLP(Qwen2_5_VLMLP):
    pass


class MiMoV2VLVisionPatchEmbed(Qwen3VLVisionPatchEmbed):
    def __init__(self, config: MiMoV2VLVisionConfig) -> None:
        super().__init__(config)
        kernel_size = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.proj = nn.Conv3d(
            self.in_channels, self.embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=False
        )


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
        # CODEPATH: sliding-window blocks only see `sliding_window` patches on each side, plus the attention sinks.
        self.window_size = (config.sliding_window, config.sliding_window) if self.is_sliding else None

        self.qkv = nn.Linear(self.dim, (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim, bias=True)
        self.proj = nn.Linear(self.num_heads * self.head_dim, self.dim)
        # The sinks are one extra logit per head in the softmax denominator.
        use_sinks = self.is_sliding and config.use_attention_sinks
        self.sinks = nn.Parameter(torch.zeros(self.num_heads)) if use_sinks else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        **kwargs,
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

        # Process each image or video frame separately.
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        splits = [torch.split(tensor, lengths, dim=2) for tensor in (query_states, key_states, value_states)]
        attn_outputs = [
            sink_attention_forward(
                self,
                query,
                key,
                value,
                attention_mask=None,
                scaling=self.scaling,
                dropout=0.0 if not self.training else self.attention_dropout,
                window_size=self.window_size,
            )[0]
            for query, key, value in zip(*splits)
        ]
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
        self.norm = MiMoV2VLRMSNorm(config.hidden_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size, bias=config.merger_bias)
        self.linear_fc2 = nn.Linear(self.hidden_size, config.out_hidden_size, bias=config.merger_bias)


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

        self.patch_embed = MiMoV2VLVisionPatchEmbed(config)
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
            # CODEPATH: permute the patches when the traversal order changes between consecutive blocks.
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


# --------------------------------------------------------------------------------------------------------------------
# Audio encoder: codec IDs -> language-model embeddings
# --------------------------------------------------------------------------------------------------------------------


class MiMoV2VLAudioRotaryEmbedding(Qwen2RotaryEmbedding):
    pass


class MiMoV2VLAudioMLP(Qwen2MLP):
    pass


class MiMoV2VLAudioAttention(Qwen2Attention):
    """Attention between the codec frames of a single group, bidirectional unless `config.is_causal`."""

    def __init__(self, config: MiMoV2VLAudioConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.is_causal = config.is_causal


class MiMoV2VLAudioEncoderLayer(Qwen2DecoderLayer):
    def __init__(self, config: MiMoV2VLAudioConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = MiMoV2VLAudioAttention(config, layer_idx)
        self.mlp = MiMoV2VLAudioMLP(config)
        self.input_layernorm = MiMoV2VLRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MiMoV2VLRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class MiMoV2VLAudioProjection(nn.Module):
    def __init__(self, config: MiMoV2VLAudioConfig):
        super().__init__()
        input_size = config.hidden_size * config.group_size
        self.linear_fc1 = nn.Linear(input_size, input_size * 4, bias=False)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(input_size * 4, config.out_hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_states)))


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
        # CODEPATH: the final norm is optional (`use_final_norm`).
        self.norm = MiMoV2VLRMSNorm(config.hidden_size, eps=config.rms_norm_eps) if config.use_final_norm else None
        self.rotary_emb = MiMoV2VLAudioRotaryEmbedding(config)
        # CODEPATH: the projection to the language model is either a single linear layer or a 2-layer MLP.
        if config.projection_layers == 1:
            self.projection = nn.Linear(config.hidden_size * config.group_size, config.out_hidden_size, bias=False)
        else:
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
        attention_mask = None
        # CODEPATH: the codec frames of a group attend bidirectionally, unless the checkpoint uses causal attention.
        if self.config.is_causal:
            attention_mask = create_causal_mask(
                config=self.config,
                inputs_embeds=hidden_states,
                attention_mask=None,
                past_key_values=None,
                position_ids=position_ids,
            )
        for layer in self.layers:
            hidden_states = layer(
                hidden_states, attention_mask=attention_mask, position_embeddings=position_embeddings, **kwargs
            )
        if self.norm is not None:
            hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPooling(
            last_hidden_state=hidden_states, pooler_output=self.projection(hidden_states.flatten(1))
        )


# --------------------------------------------------------------------------------------------------------------------
# Audio tokenizer: log-mel features -> codec IDs
# --------------------------------------------------------------------------------------------------------------------


class MiMoV2VLAudioTokenizerRotaryEmbedding(MiMoV2VLAudioRotaryEmbedding):
    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Unlike the audio encoder, the cos/sin tables stay in float32: queries and keys are rotated in float32.
        freqs = position_ids[..., None].float() * self.inv_freq.to(device=x.device, dtype=torch.float)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos() * self.attention_scaling, emb.sin() * self.attention_scaling


class MiMoV2VLAudioCodebook(nn.Module):
    """Euclidean codebook of one residual vector quantization stage."""

    def __init__(self, hidden_size: int, codebook_size: int):
        super().__init__()
        self.embed = nn.Buffer(torch.zeros(codebook_size, hidden_size))

    def encode(self, hidden_states: torch.Tensor) -> torch.LongTensor:
        """Returns the index of the closest codebook entry of each vector of `hidden_states`."""
        codebook = self.embed.float()
        distances = (
            hidden_states.square().sum(-1, keepdim=True) - 2 * hidden_states @ codebook.T + codebook.square().sum(-1)
        )
        return distances.argmin(-1)


class MiMoV2VLAudioResidualVectorQuantizer(nn.Module):
    def __init__(self, config: MiMoV2VLAudioTokenizerConfig):
        super().__init__()
        self.codebooks = nn.ModuleList(
            [MiMoV2VLAudioCodebook(config.hidden_size, codebook_size) for codebook_size in config.codebook_sizes]
        )

    def encode(self, hidden_states: torch.Tensor) -> torch.LongTensor:
        """Returns the codec IDs of shape `(..., num_codebooks)` of `hidden_states`."""
        residual = hidden_states.float()
        codes = []
        for codebook in self.codebooks:
            indices = codebook.encode(residual)
            residual = residual - F.embedding(indices, codebook.embed.float())
            codes.append(indices)
        return torch.stack(codes, dim=-1)


class MiMoV2VLAudioTokenizerAttention(WhisperAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        **kwargs,
    ) -> torch.Tensor:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_fp32(query_states, key_states, cos, sin)

        attn_output = F.scaled_dot_product_attention(
            query_states, key_states, value_states, attn_mask=attention_mask, scale=self.scaling
        )
        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        return self.out_proj(attn_output)


class MiMoV2VLAudioTokenizerLayer(GradientCheckpointingLayer):
    def __init__(self, config: MiMoV2VLAudioTokenizerConfig, layer_idx: int):
        super().__init__()
        self.self_attn = MiMoV2VLAudioTokenizerAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            config=config,
            layer_idx=layer_idx,
            is_causal=config.is_causal,
        )
        self.self_attn_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.activation_fn = ACT2FN[config.hidden_act]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask, position_embeddings)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc2(self.activation_fn(self.fc1(hidden_states)))
        hidden_states = residual + hidden_states
        return hidden_states


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
    _no_split_modules = ["MiMoV2VLAudioTokenizerLayer"]

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
            [MiMoV2VLAudioTokenizerLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
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

    def _get_attention_masks(self, lengths: torch.LongTensor, sequence_length: int, dtype: torch.dtype) -> dict:
        """Returns the additive attention mask of each layer type, hiding padding (and future frames if causal)."""
        positions = torch.arange(sequence_length, device=lengths.device)
        distance = positions[None, :] - positions[:, None]
        allowed = positions[None, None, :] < lengths[:, None, None]
        if self.config.is_causal:
            allowed = allowed & (distance <= 0)
        masks = {"full_attention": allowed}
        masks["sliding_attention"] = allowed & (distance >= -self.config.sliding_window)
        return {
            layer_type: torch.zeros(mask.shape, dtype=dtype, device=mask.device).masked_fill(~mask, -torch.inf)[
                :, None
            ]
            for layer_type, mask in masks.items()
        }

    @torch.no_grad()
    def encode_segments(self, input_features: torch.Tensor, lengths: torch.LongTensor) -> list[torch.LongTensor]:
        """Encodes a padded batch of mel segments into one `(codec_time, num_codebooks)` tensor per segment."""
        input_features = input_features.to(dtype=self.conv1.weight.dtype)
        hidden_states = F.gelu(self.conv1(input_features.transpose(1, 2)))
        hidden_states = F.gelu(self.conv2(hidden_states)).transpose(1, 2)
        lengths = self._get_conv_output_lengths(lengths.to(hidden_states.device))

        position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device)[None]
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        attention_masks = self._get_attention_masks(lengths, hidden_states.shape[1], hidden_states.dtype)
        skip_hidden_states = None
        for layer_idx, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states, attention_masks[self.config.layer_types[layer_idx]], position_embeddings
            )
            if layer_idx == self.config.skip_layer_idx:
                skip_hidden_states = hidden_states
        if skip_hidden_states is not None:
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
        # trf-ignore: TRF056
        return list(codes.split(lengths.tolist(), dim=0))

    @torch.no_grad()
    @auto_docstring
    def forward(
        self, input_features: torch.Tensor, feature_attention_mask: torch.Tensor | None = None, **kwargs
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
            # trf-ignore: TRF056
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


# --------------------------------------------------------------------------------------------------------------------
# Multimodal model
# --------------------------------------------------------------------------------------------------------------------


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
class MiMoV2VLModel(MiMoV2VLPreTrainedModel):
    def __init__(self, config: MiMoV2VLConfig):
        super().__init__(config)
        self.language_model = MiMoV2VLTextModel(config.text_config)
        self.visual = MiMoV2VLVisionModel(config.vision_config)
        self.audio_encoder = MiMoV2VLAudioEncoder(config.audio_config)
        self.audio_tokenizer = MiMoV2VLAudioTokenizerEncoder(config.audio_tokenizer_config)
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    @can_return_tuple
    @auto_docstring
    def get_image_features(
        self, pixel_values: torch.Tensor, image_grid_thw: torch.LongTensor, **kwargs: Unpack[TransformersKwargs]
    ) -> tuple | BaseModelOutputWithPooling:
        vision_outputs = self.visual(pixel_values, image_grid_thw, return_dict=True, **kwargs)
        split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        vision_outputs.pooler_output = torch.split(vision_outputs.pooler_output, split_sizes)
        return vision_outputs

    @can_return_tuple
    @auto_docstring
    def get_video_features(
        self,
        pixel_values_videos: torch.Tensor,
        video_grid_thw: torch.LongTensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        return self.get_image_features(pixel_values_videos, video_grid_thw, **kwargs)

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
class MiMoV2VLForConditionalGeneration(MiMoV2VLPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_gather_output"}

    def __init__(self, config: MiMoV2VLConfig):
        super().__init__(config)
        self.model = MiMoV2VLModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    @auto_docstring
    def get_image_features(
        self, pixel_values: torch.Tensor, image_grid_thw: torch.LongTensor, **kwargs: Unpack[TransformersKwargs]
    ) -> tuple | BaseModelOutputWithPooling:
        return self.model.get_image_features(pixel_values, image_grid_thw, **kwargs)

    @auto_docstring
    def get_video_features(
        self,
        pixel_values_videos: torch.Tensor,
        video_grid_thw: torch.LongTensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        return self.model.get_video_features(pixel_values_videos, video_grid_thw, **kwargs)

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
        model_inputs = super().prepare_inputs_for_generation(
            input_ids, is_first_iteration=is_first_iteration, **kwargs
        )
        if use_cache and not is_first_iteration:
            for name in _MULTIMODAL_INPUT_NAMES:
                model_inputs.pop(name, None)
        return model_inputs

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
        return super()._expand_inputs_for_generation(
            expand_size=expand_size, is_encoder_decoder=is_encoder_decoder, input_ids=input_ids, **model_kwargs
        )


# --------------------------------------------------------------------------------------------------------------------
# Image and video processors
# --------------------------------------------------------------------------------------------------------------------


def smart_resize(height: int, width: int, factor: int, min_pixels: int, max_pixels: int) -> tuple[int, int]:
    """Apply MiMo's minimum merge-block scaling before Qwen-style pixel-budget rounding."""
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


def resize_frames(frames: torch.Tensor, size: SizeDict, resample: PILImageResampling, factor: int) -> torch.Tensor:
    """Resize float32 frames without antialiasing, as in the original MiMo processor."""
    if not size.shortest_edge or not size.longest_edge:
        raise ValueError("size must contain shortest_edge and longest_edge pixel budgets.")
    resized_size = smart_resize(
        *frames.shape[-2:], factor=factor, min_pixels=size.shortest_edge, max_pixels=size.longest_edge
    )
    interpolation_modes = {PILImageResampling.BILINEAR: "bilinear", PILImageResampling.BICUBIC: "bicubic"}
    if resample not in interpolation_modes:
        raise ValueError("The MiMo V2 series supports bilinear and bicubic resampling.")
    frames = frames.float()
    if tuple(frames.shape[-2:]) != resized_size:
        frames = F.interpolate(
            frames, size=resized_size, mode=interpolation_modes[resample], align_corners=False, antialias=False
        )
    return frames


def normalize_frames(processor, frames, do_rescale, rescale_factor, do_normalize, image_mean, image_std):
    """Use the common backend operations with explicitly float32 source-pixel statistics."""
    frames = frames.float()
    if do_normalize:
        scale = 1 / rescale_factor if do_rescale else 1.0
        mean = torch.as_tensor(image_mean, dtype=torch.float32, device=frames.device) * scale
        std = torch.as_tensor(image_std, dtype=torch.float32, device=frames.device) * scale
        return processor.normalize(frames, mean, std)
    if do_rescale:
        return processor.rescale(frames, rescale_factor)
    return frames


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
        return resize_frames(images, size, resample, factor)

    def rescale_and_normalize(self, images, do_rescale, rescale_factor, do_normalize, image_mean, image_std):
        return normalize_frames(self, images, do_rescale, rescale_factor, do_normalize, image_mean, image_std)

    def patchify(self, images, patch_size, merge_size, temporal_patch_size):
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
    """Share the pre-padding pixel budget between preprocessing and patch-count estimation."""
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

    def _decode_and_sample_videos(self, videos, video_metadata, do_sample_frames=None, sample_indices_fn=None):
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
        """Use rounded, endpoint-exclusive intervals, retaining repeats for short videos."""
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
        frames = resize_frames(videos.flatten(0, 1), video_size, resample, factor)
        return frames.reshape(batch_size, num_frames, *frames.shape[1:])

    def rescale_and_normalize(self, images, do_rescale, rescale_factor, do_normalize, image_mean, image_std):
        return normalize_frames(self, images, do_rescale, rescale_factor, do_normalize, image_mean, image_std)

    def patchify(
        self, videos, patch_size, merge_size, temporal_patch_size, temporal_compression_ratio=1
    ) -> tuple[torch.Tensor, int, int, int]:
        if min(patch_size, temporal_patch_size, merge_size, temporal_compression_ratio) <= 0:
            raise ValueError("Patch, merge and compression sizes must be positive.")
        if any(length % (patch_size * merge_size) for length in videos.shape[-2:]):
            raise ValueError("Frame dimensions must be divisible by their temporal and spatial patch factors.")
        temporal_factor = temporal_patch_size * temporal_compression_ratio
        if pad_length := -videos.shape[1] % temporal_factor:
            videos = torch.cat((videos, videos[:, -1:].expand(-1, pad_length, -1, -1, -1)), dim=1)
        # The patches are contiguous, as the patch embedding of the vision encoder reshapes them with `view`.
        return super().patchify(videos.contiguous(), patch_size, merge_size, temporal_patch_size)

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
]
