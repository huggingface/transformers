# Copyright 2026 The Dots Studio team and the HuggingFace Inc. team. All rights reserved.
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
"""PyTorch Dots 3 Note Preview model for Hugging Face Transformers."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict
from torch import nn

from ... import initialization as init
from ...audio_utils import make_list_of_audio
from ...cache_utils import Cache, DynamicCache
from ...configuration_utils import PreTrainedConfig, SubConfigSpec
from ...feature_extraction_utils import BatchFeature
from ...masking_utils import create_bidirectional_mask, create_causal_mask, create_sliding_window_causal_mask
from ...modeling_outputs import (
    BaseModelOutputWithPooling,
    MoeCausalLMOutputWithPast,
    MoeModelOutputWithPast,
)
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...utils import (
    TensorType,
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
    is_torch_available,  # noqa: F401 -- used by the inherited feature extractor
    logging,
)
from ...utils.generic import (
    accepts_precomputed_kwargs,
    get_max_seqlen,
    is_flash_attention_requested,
    merge_with_config_defaults,
)
from ...utils.import_utils import requires
from ...utils.output_capturing import capture_outputs
from ...vision_utils import get_vision_attention_seqlens, get_vision_position_ids
from ..deepseek_v3.modeling_deepseek_v3 import DeepseekV3Attention, eager_attention_forward
from ..deepseek_v32.configuration_deepseek_v32 import DeepseekV32Config
from ..deepseek_v32.modeling_deepseek_v32 import (
    DeepseekV32DecoderLayer,
    DeepseekV32Experts,
    DeepseekV32ForCausalLM,
    DeepseekV32MLP,
    DeepseekV32Model,
    DeepseekV32MoE,
    DeepseekV32PreTrainedModel,
    DeepseekV32RMSNorm,
    DeepseekV32TopkRouter,
)
from ..diffusion_gemma.modeling_diffusion_gemma import DiffusionGemmaTextRotaryEmbedding
from ..evolla.modeling_evolla import EvollaFeedForward
from ..glm_image.modeling_glm_image import GlmImageVisionPatchEmbed
from ..glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaIndexer, apply_rotary_pos_emb_interleave
from ..glm_ocr.modeling_glm_ocr import GlmOcrVisionAttention, apply_rotary_pos_emb_vision
from ..glmasr.configuration_glmasr import GlmAsrEncoderConfig
from ..hy_v3.modeling_hy_v3 import HYV3TopKRouter
from ..inkling.modeling_inkling import InklingModel
from ..llama.modeling_llama import LlamaMLP
from ..mimo_v2_flash.modeling_mimo_v2_flash import MiMoV2FlashMoE
from ..nemotron.modeling_nemotron import NemotronAttention
from ..nemotron_asr_streaming.modeling_nemotron_asr_streaming import _mask_subsampled_frames
from ..phi.modeling_phi import PhiRotaryEmbedding
from ..phi3.modeling_phi3 import Phi3DecoderLayer, Phi3MLP, Phi3PreTrainedModel
from ..qwen2_vl.modeling_qwen2_vl import (
    PatchMerger,
    Qwen2VisionTransformerPretrainedModel,
    Qwen2VLPreTrainedModel,
    Qwen2VLVisionBlock,
    Qwen2VLVisionRotaryEmbedding,
)
from ..qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor
from ..video_llama_3.configuration_video_llama_3 import VideoLlama3VisionConfig
from ..voxtral.modeling_voxtral import VoxtralForConditionalGeneration
from ..whisper.feature_extraction_whisper import WhisperFeatureExtractor


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="dots-studio/dots3-note-prev")
@strict
class Dots3NoteVisionConfig(VideoLlama3VisionConfig):
    r"""
    embed_dim (`int`, *optional*, defaults to 1536):
        Width of the patch embeddings and transformer blocks.
    mlp_layer_types (`list[str]`, *optional*):
        Dense or sparse MLP for each layer. Expert counts are specified through `per_layer_config`.
    router_scaling_factor (`float`, *optional*, defaults to 1.0):
        Scale applied to the vision router scores.
    adapter_in_dim (`int`, *optional*, defaults to 1536):
        Input width of the vision adapter. It must match `embed_dim`.
    adapter_out_dim (`int`, *optional*, defaults to 5120):
        Output width of the vision adapter. It must match the text hidden size.
    adapter_merge_size (`int`, *optional*, defaults to 2):
        Spatial merge factor used by the vision adapter.
    """

    model_type = "dots3_note_vision_encoder"
    layer_norm_eps = AttributeError()

    embed_dim: int = 1536
    hidden_size: int = 1536
    intermediate_size: int = 4224
    moe_intermediate_size: int = 2112
    num_hidden_layers: int = 42
    num_attention_heads: int = 24
    patch_size: int = 14
    spatial_merge_size: int = 2
    rms_norm_eps: float = 1e-5
    attention_bias: bool = False
    mlp_bias: bool = False
    hidden_act: str = "silu"
    mlp_layer_types: list[str] | None = None
    num_local_experts: int = 64
    num_experts_per_tok: int | None = None
    router_scaling_factor: float | None = None
    adapter_in_dim: int = 1536
    adapter_out_dim: int = 5120
    adapter_merge_size: int = 2

    def __post_init__(self, **kwargs):
        pyramid_num_routed = kwargs.pop("pyramid_num_routed", None)
        if pyramid_num_routed is None and self.mlp_layer_types is None:
            pyramid_num_routed = [-1] * 25 + list(range(4, 65, 4)) + [64]
        if self.num_experts_per_tok is None:
            self.num_experts_per_tok = kwargs.get("capacity_factor", 2)
        if self.router_scaling_factor is None:
            self.router_scaling_factor = kwargs.get("router_scale", 1.0)
        if pyramid_num_routed is not None:
            self.mlp_layer_types = ["sparse" if count > 0 else "dense" for count in pyramid_num_routed]
            kwargs.setdefault(
                "per_layer_config",
                {
                    i: {"num_local_experts": count, "num_experts_per_tok": min(self.num_experts_per_tok, count)}
                    for i, count in enumerate(pyramid_num_routed)
                    if count > 0
                },
            )
        self.hidden_size = self.embed_dim
        self.num_attention_heads = kwargs.pop("num_heads", self.num_attention_heads)
        self.attention_bias = kwargs.pop("use_bias", self.attention_bias)
        kwargs.pop("is_causal", None)
        PreTrainedConfig.__post_init__(self, **kwargs)

    def validate_architecture(self):
        if self.embed_dim % self.num_attention_heads != 0:
            raise ValueError("embed_dim must be divisible by num_attention_heads")
        if self.adapter_in_dim != self.embed_dim:
            raise ValueError("adapter_in_dim must match embed_dim")
        if self.mlp_bias and "sparse" in self.mlp_layer_types:
            raise ValueError("Dots3 vision stacked experts require mlp_bias=False")
        if self.adapter_merge_size != self.spatial_merge_size:
            raise ValueError("adapter_merge_size must match spatial_merge_size")


@auto_docstring(checkpoint="dots-studio/dots3-note-prev")
@strict
class Dots3NoteAudioConfig(GlmAsrEncoderConfig):
    r"""
    feature_size (`int`, *optional*, defaults to 128):
        Number of log-mel filter-bank channels.
    hop_length (`int`, *optional*, defaults to 160):
        Number of waveform samples between adjacent log-mel frames.
    downsample_hidden_size (`int`, *optional*, defaults to 480):
        Hidden width of the temporal downsampling projection.
    adapter_input_size (`int`, *optional*, defaults to 1280):
        Input width of the audio-to-text adapter.
    adapter_output_size (`int`, *optional*, defaults to 5120):
        Output width of the audio-to-text adapter. It must match the text hidden size.
    """

    model_type = "dots3_note_audio_encoder"
    base_config_key = "audio_config"
    num_mel_bins = AttributeError()
    initializer_range = AttributeError()

    max_position_embeddings: int = 6000
    dropout: float = 0.0
    sampling_rate: int = 16_000
    feature_size: int = 128
    hop_length: int = 160
    downsample_hidden_size: int = 480
    head_dim: int | None = None
    attention_bias: bool = True
    hidden_act: str = "silu"
    adapter_input_size: int = 1280
    adapter_output_size: int = 5120

    def __post_init__(self, **kwargs):
        self.adapter_input_size = kwargs.pop("whisper_adapter_in_dim", self.adapter_input_size)
        self.adapter_output_size = kwargs.pop("whisper_adapter_out_dim", self.adapter_output_size)
        # Released checkpoints store encoder dimensions in a Whisper-style dictionary.
        whisper_config = kwargs.pop("whisper_config", None) or {}
        for legacy_name, name in {
            "d_model": "hidden_size",
            "encoder_attention_heads": "num_attention_heads",
            "encoder_layers": "num_hidden_layers",
            "encoder_ffn_dim": "intermediate_size",
            "num_mel_bins": "feature_size",
            "max_source_positions": "max_position_embeddings",
        }.items():
            if legacy_name in whisper_config:
                setattr(self, name, whisper_config[legacy_name])
        self.dropout = whisper_config.get("dropout", self.dropout)
        self.attention_dropout = whisper_config.get("attention_dropout", self.attention_dropout)
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.num_key_value_heads = self.num_attention_heads
        self.conv_output_size = self.downsample_hidden_size * ((self.feature_size + 7) // 8)
        self.samples_per_token = self.hop_length * 8
        kwargs.setdefault("partial_rotary_factor", 0.5)
        PreTrainedConfig.__post_init__(self, **kwargs)

    def validate_architecture(self):
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("audio hidden_size must be divisible by num_attention_heads")
        if self.adapter_input_size != self.hidden_size:
            raise ValueError("adapter_input_size must match hidden_size")
        if min(self.sampling_rate, self.feature_size, self.hop_length) <= 0:
            raise ValueError("audio sampling and feature dimensions must be positive")


@auto_docstring(checkpoint="dots-studio/dots3-note-prev")
@strict
class Dots3NoteTextConfig(DeepseekV32Config):
    r"""
    n_group (`int`, *optional*, defaults to 1):
        Number of groups into which routed experts are partitioned during selection.
    mlp_layer_types (`list[str]`, *optional*):
        Per-layer `"dense"` or `"sparse"` MLP types, derived from the released MoE schedule when omitted.
    index_topk (`int`, *optional*, defaults to 2048):
        Number of key positions selected by dynamic sparse attention.
    index_head_dim (`int`, *optional*, defaults to 128):
        Per-head width of the dynamic sparse attention indexer.
    index_n_heads (`int`, *optional*, defaults to 64):
        Number of heads in the dynamic sparse attention indexer.
    shared_experts_intermediate_size (`int`, *optional*, defaults to 1536):
        Intermediate width of each shared expert.
    """

    first_k_dense_replace = AttributeError()
    base_model_pp_plan = AttributeError()
    model_type = "dots3_note_text"
    base_model_tp_plan = {
        "layers.*.self_attn.q_b_proj": "colwise",
        "layers.*.self_attn.kv_a_proj_with_mqa": "mla_kv_a_proj",
        "layers.*.self_attn.kv_b_proj": "colwise",
        "layers.*.self_attn.g_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
        "layers.*.mlp.experts.down_proj": "rowwise",
        "layers.*.mlp.experts": "moe_tp_experts",
        "layers.*.mlp.shared_experts.gate_proj": "colwise",
        "layers.*.mlp.shared_experts.up_proj": "colwise",
        "layers.*.mlp.shared_experts.down_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    default_theta = 80_000_000.0
    base_config_key = "text_config"

    vocab_size: int = 152064
    hidden_size: int = 5120
    intermediate_size: int = 13824
    num_hidden_layers: int = 46
    max_position_embeddings: int = 524288
    rms_norm_eps: float = 1e-5
    pad_token_id: int | None = 151659
    bos_token_id: int | None = 151643
    eos_token_id: int | list[int] | None = 151668
    head_dim: int = 256

    q_lora_rank: int | None = 1024

    sliding_window: int | None = None

    moe_intermediate_size: int = 1536
    shared_experts_intermediate_size: int = 1536
    routed_scaling_factor: float = 1.0
    n_group: int = 1
    topk_group: int = 1

    def __post_init__(self, **kwargs):
        first_k_dense_replace = kwargs.pop("first_k_dense_replace", 1)
        moe_layer_freq = kwargs.pop("moe_layer_freq", 1)
        # Legacy checkpoints call DSA layers "full_attention"; canonical configs also store per_layer_config.
        use_dsa = kwargs.pop("use_dsa", None if "per_layer_config" in kwargs else True)
        use_sliding_window = kwargs.pop("use_sliding_window", True)
        if self.mlp_layer_types is None:
            self.mlp_layer_types = [
                "sparse"
                if i >= first_k_dense_replace
                and (bool(moe_layer_freq[i]) if isinstance(moe_layer_freq, (list, tuple)) else i % moe_layer_freq == 0)
                else "dense"
                for i in range(self.num_hidden_layers)
            ]
        self.shared_experts_intermediate_size = kwargs.pop(
            "moe_shared_expert_intermediate_size", self.shared_experts_intermediate_size
        )
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.sliding_window is None:
            self.sliding_window = kwargs.pop("sliding_window_size", 512)

        if self.layer_types is None:
            full_attention_type = "full_attention" if use_dsa is False else "indexed_attention"
            self.layer_types = [
                full_attention_type if not use_sliding_window or i < 2 or i % 4 == 1 else "sliding_attention"
                for i in range(self.num_hidden_layers)
            ]
        else:
            self.layer_types = list(self.layer_types)
        self.layer_types = [
            "indexed_attention"
            if use_dsa is True and layer_type == "full_attention"
            else "full_attention"
            if use_dsa is False and layer_type == "indexed_attention"
            else layer_type
            for layer_type in self.layer_types
        ]

        swa_rope_theta = kwargs.pop("swa_rope_theta", 50_000.0)
        rope_theta = kwargs.pop("rope_theta", self.default_theta)
        if self.rope_parameters is None:
            self.rope_parameters = kwargs.pop("rope_scaling", None) or {
                layer_type: {
                    "rope_type": "default",
                    "rope_theta": swa_rope_theta if layer_type == "sliding_attention" else rope_theta,
                }
                for layer_type in set(self.layer_types)
            }
        if not self.nested_rope_parameter_keys(self.rope_parameters):
            self.rope_parameters = {layer_type: dict(self.rope_parameters) for layer_type in set(self.layer_types)}
        # Normalize the released checkpoint's SWA aliases into the shared per-layer configuration.
        sliding_config = {
            name: kwargs.pop(f"swa_{name}", default)
            for name, default in {
                "num_attention_heads": 64,
                "num_key_value_heads": 64,
                "q_lora_rank": 1024,
                "kv_lora_rank": 1024,
                "head_dim": 256,
                "qk_nope_head_dim": 192,
                "v_head_dim": 128,
            }.items()
        }
        sliding_config["qk_rope_head_dim"] = kwargs.pop(
            "swa_qk_rope_head_dim", sliding_config["head_dim"] - sliding_config["qk_nope_head_dim"]
        )
        kwargs.setdefault(
            "per_layer_config",
            {i: sliding_config for i, layer_type in enumerate(self.layer_types) if layer_type == "sliding_attention"},
        )
        PreTrainedConfig.__post_init__(self, **kwargs)

    def convert_rope_params_to_dict(self, **kwargs):
        kwargs = PreTrainedConfig.convert_rope_params_to_dict(self, **kwargs)
        per_layer_config = {int(i): values for i, values in (kwargs.get("per_layer_config") or {}).items()}
        for layer_type in set(self.layer_types):
            layer_config = per_layer_config.get(self.layer_types.index(layer_type), {})
            self.rope_parameters[layer_type]["partial_rotary_factor"] = layer_config.get(
                "qk_rope_head_dim", self.qk_rope_head_dim
            ) / layer_config.get("head_dim", self.head_dim)
        return kwargs

    def validate_architecture(self):
        if any(self.per_layer_config[layer_type].q_lora_rank is None for layer_type in set(self.layer_types)):
            raise ValueError("Dots 3 Note Preview requires q_lora_rank for MLA queries")
        if self.n_shared_experts is None or self.n_shared_experts < 1:
            raise ValueError("Dots 3 Note Preview requires at least one shared expert")
        if self.n_group < 1 or self.n_routed_experts % self.n_group != 0:
            raise ValueError("n_group must evenly divide n_routed_experts")
        if self.n_routed_experts // self.n_group < 2:
            raise ValueError("Grouped routing requires at least two experts per group")
        if not 1 <= self.topk_group <= self.n_group:
            raise ValueError("topk_group must be in [1, n_group]")
        for layer_type in set(self.layer_types):
            config = self.per_layer_config[layer_type]
            if config.num_key_value_heads != config.num_attention_heads:
                raise ValueError("num_key_value_heads must match num_attention_heads")
            if layer_type == "indexed_attention" and config.qk_rope_head_dim > self.index_head_dim:
                raise ValueError("qk_rope_head_dim must not exceed index_head_dim")
        if "sliding_attention" in self.layer_types:
            config = self.per_layer_config["sliding_attention"]
            if config.head_dim != config.qk_nope_head_dim + config.qk_rope_head_dim:
                raise ValueError("SWA head_dim must equal qk_nope_head_dim + qk_rope_head_dim")


@auto_docstring(checkpoint="dots-studio/dots3-note-prev")
@strict
class Dots3NoteConfig(PreTrainedConfig):
    model_type = "dots3_note"
    sub_configs_defaults = {
        "text_config": SubConfigSpec(config_class=Dots3NoteTextConfig),
        "vision_config": SubConfigSpec(config_class=Dots3NoteVisionConfig, optional=True),
        "audio_config": SubConfigSpec(config_class=Dots3NoteAudioConfig, optional=True),
    }
    keys_to_ignore_at_inference = ["past_key_values"]

    text_config: dict | Dots3NoteTextConfig | None = None
    vision_config: dict | Dots3NoteVisionConfig | None = None
    audio_config: dict | Dots3NoteAudioConfig | None = None
    image_token_id: int = 151660
    video_token_id: int = 151680
    audio_token_id: int = 151720
    tie_word_embeddings: bool = False

    def __post_init__(self, **kwargs):
        # Published checkpoints store the text configuration at the top level.
        if self.text_config is None:
            kwargs.pop("model_type", None)
            self.text_config = Dots3NoteTextConfig(**kwargs, tie_word_embeddings=self.tie_word_embeddings)
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        if self.vision_config is not None and self.vision_config.adapter_out_dim != self.text_config.hidden_size:
            raise ValueError("vision adapter output width must match the text hidden size")
        if self.audio_config is not None and self.audio_config.adapter_output_size != self.text_config.hidden_size:
            raise ValueError("audio adapter output width must match the text hidden size")


# Text decoder
class Dots3NoteRMSNorm(DeepseekV32RMSNorm):
    pass


class Dots3NoteTextRotaryEmbedding(DiffusionGemmaTextRotaryEmbedding):
    pass


class Dots3NoteTextIndexer(GlmMoeDsaIndexer):
    # TODO: Add FP8 support through the shared indexer and quantization path.
    pass


class Dots3NoteTextMLP(DeepseekV32MLP):
    pass


class Dots3NoteTextTopkRouter(DeepseekV32TopkRouter):
    pass


class Dots3NoteExperts(DeepseekV32Experts):
    pass


class Dots3NoteTextMoE(DeepseekV32MoE):
    def __init__(self, config):
        super().__init__(config)
        self.experts = Dots3NoteExperts(config)
        self.shared_experts = Dots3NoteTextMLP(
            config, intermediate_size=config.shared_experts_intermediate_size * config.n_shared_experts
        )


def apply_rotary_pos_emb_text(q, k, cos, sin):
    q_rot, k_rot = apply_rotary_pos_emb_interleave(q, k, cos, sin)
    return q_rot.to(q.dtype), k_rot.to(k.dtype)


class Dots3NoteTextAttention(DeepseekV3Attention):
    def __init__(self, config: Dots3NoteTextConfig, layer_idx: int):
        model_config = config
        config = config.per_layer_config[layer_idx]
        super().__init__(config, layer_idx)
        self.config = model_config
        self.scaling = self.qk_head_dim**-0.5
        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None

        self.q_a_layernorm = Dots3NoteRMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.kv_a_layernorm = Dots3NoteRMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.k_rope_only_layernorm = Dots3NoteRMSNorm(self.qk_rope_head_dim, eps=config.rms_norm_eps)
        self.q_lora_scale = (self.hidden_size / self.q_lora_rank) ** 0.5
        self.kv_lora_scale = (self.hidden_size / self.kv_lora_rank) ** 0.5
        self.g_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False)

        self.indexer = (
            Dots3NoteTextIndexer(config, layer_idx) if config.layer_types[layer_idx] == "indexed_attention" else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        batch_size, seq_length = hidden_states.shape[:-1]
        query_shape = (batch_size, seq_length, -1, self.qk_head_dim)

        q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))
        # Dots rescales the normalized Q/KV latents before projection, including the indexer's query input.
        q_resid = q_resid * self.q_lora_scale
        q_states = self.q_b_proj(q_resid).view(query_shape).transpose(1, 2)
        q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        kv_pass, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pass = self.kv_a_layernorm(kv_pass.contiguous()).view(batch_size, 1, seq_length, self.kv_lora_rank)
        k_pass = k_pass * self.kv_lora_scale

        # Dots additionally normalizes the shared K-RoPE head before applying interleaved RoPE.
        k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)
        k_rot = self.k_rope_only_layernorm(k_rot)

        cos, sin = position_embeddings
        q_rot, k_rot = apply_rotary_pos_emb_text(q_rot, k_rot, cos, sin)

        if past_key_values is not None:
            k_pass, k_rot = past_key_values.update(k_pass, k_rot, self.layer_idx)

        query_states = torch.cat((q_pass, q_rot), dim=-1)
        key_states, value_states = self.expand_kv(k_pass, k_rot)

        sparse_indices = None
        if self.indexer is not None:
            topk_indices = self.indexer(
                hidden_states,
                q_resid,
                position_embeddings,
                attention_mask[:, 0],
                position_ids=None,  # Unused backward-compatibility argument in the shared indexer.
                past_key_values=past_key_values,
            )
            if self.config._attn_implementation in ("eager", "sdpa"):
                index_mask = (
                    topk_indices.new_ones((batch_size, seq_length, key_states.shape[-2]), dtype=torch.bool)
                    .scatter(-1, topk_indices.long(), False)
                    .unsqueeze(1)
                )
                if attention_mask.dtype == torch.bool:
                    attention_mask = attention_mask & ~index_mask
                else:
                    attention_mask = attention_mask.masked_fill(index_mask, torch.finfo(hidden_states.dtype).min)
            else:
                sparse_indices = topk_indices

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            indices=sparse_indices,
            **kwargs,
        )

        # Dots applies a per-head sigmoid gate before the output projection.
        gate = self.g_proj(hidden_states).view(batch_size, seq_length, -1, 1)
        attn_output = attn_output * gate.sigmoid()

        attn_output = attn_output.reshape(batch_size, seq_length, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Dots3NoteTextDecoderLayer(DeepseekV32DecoderLayer):
    def __init__(self, config: Dots3NoteTextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.input_layernorm = Dots3NoteRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = Dots3NoteRMSNorm(config.hidden_size, config.rms_norm_eps)


@auto_docstring
class Dots3NotePreTrainedModel(DeepseekV32PreTrainedModel):
    config: Dots3NoteConfig
    config_class = Dots3NoteConfig
    _no_split_modules = ["Dots3NoteAudioLayer", "Dots3NoteTextDecoderLayer", "Dots3NoteVisionBlock"]
    _keep_in_fp32_modules = AttributeError()
    _keys_to_ignore_on_load_unexpected = [
        r"^model\.(language_model\.)?layers\.46\.",
        r"^model\.(language_model\.)?mtp\.",
    ]

    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, Dots3NoteTextTopkRouter):
            init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            init.zeros_(module.e_score_correction_bias)
        elif isinstance(module, Dots3NoteExperts):
            init.normal_(module.gate_up_proj, mean=0.0, std=self.config.initializer_range)
            init.normal_(module.down_proj, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, Dots3NoteTextRotaryEmbedding):
            for layer_type, rope_init_fn in module.rope_init_fns.items():
                inv_freq, _ = rope_init_fn(module.config.per_layer_config[layer_type], layer_type=layer_type)
                init.copy_(getattr(module, f"{layer_type}_inv_freq"), inv_freq)
                init.copy_(getattr(module, f"{layer_type}_original_inv_freq"), inv_freq)


class Dots3NoteTextModel(Dots3NotePreTrainedModel, DeepseekV32Model):
    config_class = Dots3NoteTextConfig

    def __init__(self, config: Dots3NoteTextConfig):
        super().__init__(config)
        self.norm = Dots3NoteRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeModelOutputWithPast:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_len = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = (torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_len).unsqueeze(0)

        position_embeddings = {
            layer_type: self.rotary_emb(
                inputs_embeds.float() if layer_type == "indexed_attention" else inputs_embeds,
                position_ids,
                layer_type,
            )
            for layer_type in set(self.config.layer_types)
        }

        if not isinstance(causal_masks := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            mask_functions = {
                "full_attention": lambda: create_causal_mask(**mask_kwargs),
                "sliding_attention": lambda: create_sliding_window_causal_mask(**mask_kwargs),
                "indexed_attention": lambda: create_causal_mask(**mask_kwargs, allow_is_causal_skip=False),
            }
            causal_masks = {layer_type: mask_functions[layer_type]() for layer_type in set(self.config.layer_types)}

        hidden_states = inputs_embeds
        for layer_idx, layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            layer_mask = causal_masks[self.config.layer_types[layer_idx]]
            hidden_states = layer(
                hidden_states,
                attention_mask=layer_mask,
                position_embeddings=position_embeddings[self.config.layer_types[layer_idx]],
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class Dots3NoteForCausalLM(Dots3NotePreTrainedModel, DeepseekV32ForCausalLM):
    config_class = Dots3NoteTextConfig

    def forward(self, **super_kwargs):
        r"""
        Example:

        ```python
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer

        >>> model_id = "dots-studio/dots3-note-prev"
        >>> tokenizer = AutoTokenizer.from_pretrained(model_id)
        >>> model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
        >>> inputs = tokenizer("Briefly introduce yourself.", return_tensors="pt").to(model.device)
        >>> output = model.generate(**inputs, max_new_tokens=32)
        >>> text = tokenizer.batch_decode(output, skip_special_tokens=True)
        ```
        """
        return super().forward(**super_kwargs)


# Audio encoder and adapter
class Dots3NoteAudioRotaryEmbedding(PhiRotaryEmbedding):
    pass


class Dots3NoteAudioAttention(NemotronAttention):
    def __init__(self, config: Dots3NoteAudioConfig):
        super().__init__(config)
        del self.layer_idx
        del self.partial_rotary_factor
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.is_causal = False


class Dots3NoteAudioMLP(Phi3MLP):
    def __init__(self, config: Dots3NoteAudioConfig):
        super().__init__(config)
        self.gate_up_proj = nn.Linear(config.hidden_size, 2 * config.intermediate_size)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size)


class Dots3NoteAudioLayer(Phi3DecoderLayer):
    def __init__(self, config: Dots3NoteAudioConfig):
        nn.Module.__init__(self)
        hidden_size = config.hidden_size
        self.self_attn = Dots3NoteAudioAttention(config)
        self.input_layernorm = Dots3NoteRMSNorm(hidden_size)
        self.mlp = Dots3NoteAudioMLP(config)
        self.post_attention_layernorm = Dots3NoteRMSNorm(hidden_size)
        self.resid_attn_dropout = nn.Dropout(config.dropout)
        self.resid_mlp_dropout = nn.Dropout(config.dropout)


class Dots3NoteAudioConvStem(nn.Module):
    def __init__(self, config: Dots3NoteAudioConfig):
        super().__init__()
        hidden_size = config.hidden_size
        downsample_size = config.downsample_hidden_size
        self.convs = nn.ModuleList(
            nn.Conv2d(in_channels, downsample_size, 3, stride=2, padding=1)
            for in_channels in (1, downsample_size, downsample_size)
        )
        self.conv_out = nn.Linear(config.conv_output_size, hidden_size, bias=False)
        self.hop_length = config.hop_length

    def forward(self, input_features: torch.Tensor, audio_sample_lengths: torch.Tensor) -> torch.Tensor:
        hidden_states = input_features.unsqueeze(1)
        valid_lengths = audio_sample_lengths // self.hop_length
        hidden_states = _mask_subsampled_frames(hidden_states.transpose(-1, -2), valid_lengths).transpose(-1, -2)
        for conv in self.convs:
            hidden_states = F.gelu(conv(hidden_states))
            valid_lengths = (valid_lengths + 1) // 2
            hidden_states = _mask_subsampled_frames(hidden_states.transpose(-1, -2), valid_lengths).transpose(-1, -2)
        batch_size, channels, frequency, time = hidden_states.shape
        hidden_states = hidden_states.permute(0, 3, 1, 2).reshape(batch_size, time, channels * frequency)
        return self.conv_out(hidden_states)


@auto_docstring
class Dots3NoteAudioPreTrainedModel(Phi3PreTrainedModel):
    config: Dots3NoteAudioConfig
    config_class = Dots3NoteAudioConfig
    main_input_name = "input_features"
    _no_split_modules = ["Dots3NoteAudioLayer"]
    _version = AttributeError()
    _can_record_outputs = {"hidden_states": Dots3NoteAudioLayer, "attentions": Dots3NoteAudioAttention}


@auto_docstring
class Dots3NoteAudioModel(Dots3NoteAudioPreTrainedModel):
    """Bidirectional audio encoder with a convolutional stem and rotary positions."""

    def __init__(self, config: Dots3NoteAudioConfig):
        super().__init__(config)
        self.conv_stem = Dots3NoteAudioConvStem(config)
        self.rotary_embedding = Dots3NoteAudioRotaryEmbedding(config)
        self.layers = nn.ModuleList([Dots3NoteAudioLayer(config) for _ in range(config.num_hidden_layers)])
        self.layer_norm = Dots3NoteRMSNorm(config.hidden_size)
        self.dropout = config.dropout
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_features: torch.Tensor,
        chunk_sample_lengths: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling:
        r"""
        chunk_sample_lengths (`torch.LongTensor` of shape `(num_chunks,)`):
            Number of waveform samples in each audio chunk.
        """
        if input_features.shape[0] != chunk_sample_lengths.shape[0]:
            raise ValueError("input_features and chunk_sample_lengths must contain the same number of audio chunks")
        inputs_embeds = self.conv_stem(input_features, chunk_sample_lengths)
        # The conv stem may run on a different device from the incoming chunk lengths.
        token_lengths = (
            chunk_sample_lengths.to(inputs_embeds.device) + self.config.samples_per_token - 1
        ) // self.config.samples_per_token
        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)[None, :]
        padding_mask = position_ids < token_lengths[:, None]
        position_embeddings = self.rotary_embedding(inputs_embeds, position_ids)
        hidden_states = F.dropout(inputs_embeds, p=self.dropout, training=self.training)
        attention_mask = create_bidirectional_mask(self.config, hidden_states, padding_mask)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        hidden_states = self.layer_norm(hidden_states)

        return BaseModelOutputWithPooling(
            last_hidden_state=hidden_states, pooler_output=hidden_states[padding_mask.to(hidden_states.device)]
        )


class Dots3NoteAudioAdapter(EvollaFeedForward):
    def __init__(self, config: Dots3NoteAudioConfig):
        nn.Module.__init__(self)
        self.norm = nn.LayerNorm(config.adapter_input_size)
        self.fc1 = nn.Linear(config.adapter_input_size, config.adapter_output_size)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(config.adapter_output_size, config.adapter_output_size)


# Vision encoder and adapter
class Dots3NoteVisionRotaryEmbedding(Qwen2VLVisionRotaryEmbedding):
    pass


class Dots3NoteVisionPatchEmbed(GlmImageVisionPatchEmbed):
    def __init__(self, config: Dots3NoteVisionConfig):
        super().__init__(config)
        self.in_channels = config.num_channels
        self.norm = Dots3NoteRMSNorm(self.embed_dim, eps=config.rms_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        pixel_values = pixel_values.reshape(
            -1,
            self.in_channels,
            self.patch_size,
            self.patch_size,
        )
        hidden_states = self.proj(pixel_values.to(self.proj.weight.dtype)).reshape(-1, self.embed_dim)
        return self.norm(hidden_states)


class Dots3NoteVisionMLP(LlamaMLP):
    pass


class Dots3NoteVisionTopKRouter(HYV3TopKRouter):
    pass


class Dots3NoteVisionMoE(MiMoV2FlashMoE):
    """Routed vision experts without shared experts."""

    def __init__(self, config: Dots3NoteVisionConfig, layer_idx: int):
        nn.Module.__init__(self)
        layer_config = config.per_layer_config[layer_idx]
        self.experts = Dots3NoteExperts(layer_config)
        self.gate = Dots3NoteVisionTopKRouter(layer_config)
        self.e_score_correction_bias = nn.Buffer(torch.zeros(layer_config.num_local_experts, dtype=torch.float32))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        super().forward(hidden_states)
        _, topk_weights, topk_indices = self.gate(hidden_states, self.e_score_correction_bias)
        hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])


class Dots3NoteVisionAttention(GlmOcrVisionAttention):
    def __init__(self, config: Dots3NoteVisionConfig):
        super().__init__(config)
        self.num_heads = config.num_attention_heads
        self.is_causal = False
        self.q_norm = Dots3NoteRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Dots3NoteRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        max_seqlen: int | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, *position_embeddings)
        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attention_kwargs = {
            "attention_mask": None,
            "scaling": self.scaling,
            "dropout": self.attention_dropout if self.training else 0.0,
            "is_causal": False,
            **kwargs,
        }
        if is_flash_attention_requested(self.config):
            max_seqlen = get_max_seqlen(cu_seqlens, self.config, kwargs={"max_seqlen": max_seqlen})
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                cu_seq_lens_q=cu_seqlens,
                cu_seq_lens_k=cu_seqlens,
                max_length_q=max_seqlen,
                max_length_k=max_seqlen,
                **attention_kwargs,
            )
        else:
            lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
            for length in lengths:
                # Each frame contains at least one complete spatial merge block.
                torch._check(length >= self.config.spatial_merge_size**2)
            splits = [torch.split(tensor, lengths, dim=2) for tensor in (query_states, key_states, value_states)]
            outputs = [attention_interface(self, q, k, v, **attention_kwargs) for q, k, v in zip(*splits)]
            attn_output = torch.cat([output[0] for output in outputs], dim=1)
            attn_weights = [output[1] for output in outputs] if outputs[0][1] is not None else None

        return self.proj(attn_output.reshape(seq_length, -1).contiguous()), attn_weights


class Dots3NoteVisionBlock(Qwen2VLVisionBlock):
    def __init__(self, config: Dots3NoteVisionConfig, layer_idx: int):
        nn.Module.__init__(self)
        self.norm1 = Dots3NoteRMSNorm(config.embed_dim, eps=config.rms_norm_eps)
        self.attn = Dots3NoteVisionAttention(config)
        self.norm2 = Dots3NoteRMSNorm(config.embed_dim, eps=config.rms_norm_eps)
        layer_config = config.per_layer_config[layer_idx]
        self.mlp = (
            Dots3NoteVisionMoE(config, layer_idx)
            if config.mlp_layer_types[layer_idx] == "sparse"
            else Dots3NoteVisionMLP(layer_config)
        )

    def forward(self, hidden_states: torch.Tensor, **kwargs: Unpack[TransformersKwargs]) -> torch.Tensor:
        attn_output, _ = self.attn(self.norm1(hidden_states), **kwargs)
        hidden_states = hidden_states + attn_output
        return hidden_states + self.mlp(self.norm2(hidden_states))


class Dots3NoteVisionAdapter(PatchMerger):
    pass


@auto_docstring
class Dots3NoteVisionPreTrainedModel(Qwen2VLPreTrainedModel):
    config: Dots3NoteVisionConfig
    config_class = Dots3NoteVisionConfig
    main_input_name = "pixel_values"
    input_modalities = ("image", "video")
    _no_split_modules = ["Dots3NoteVisionBlock"]

    @torch.no_grad()
    def _init_weights(self, module: nn.Module):
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, Dots3NoteVisionTopKRouter):
            init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, Dots3NoteVisionMoE):
            init.zeros_(module.e_score_correction_bias)
        elif isinstance(module, Dots3NoteExperts):
            init.normal_(module.gate_up_proj, mean=0.0, std=self.config.initializer_range)
            init.normal_(module.down_proj, mean=0.0, std=self.config.initializer_range)


@auto_docstring
class Dots3NoteVisionModel(Dots3NoteVisionPreTrainedModel, Qwen2VisionTransformerPretrainedModel):
    _can_record_outputs = {"hidden_states": Dots3NoteVisionBlock, "attentions": Dots3NoteVisionAttention}

    def __init__(self, config: Dots3NoteVisionConfig):
        super().__init__(config)
        self.patch_embed = Dots3NoteVisionPatchEmbed(config)
        self.blocks = nn.ModuleList(
            [Dots3NoteVisionBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.post_trunk_norm = Dots3NoteRMSNorm(config.embed_dim, eps=config.rms_norm_eps)
        del self.merger
        self.adapter = Dots3NoteVisionAdapter(
            dim=config.adapter_out_dim,
            context_dim=config.adapter_in_dim,
            spatial_merge_size=config.adapter_merge_size,
        )

    def get_dtype(self):
        raise AttributeError("Use the standard dtype property")

    def get_device(self):
        raise AttributeError("Use the standard device property")

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor, **kwargs) -> BaseModelOutputWithPooling:
        r"""
        grid_thw (`torch.Tensor`):
            Temporal, height, and width patch-grid dimensions for each input.
        """
        position_ids = get_vision_position_ids(grid_thw, self.spatial_merge_size, kwargs=kwargs)
        cu_seqlens, max_seqlen = get_vision_attention_seqlens(grid_thw, self.config, kwargs=kwargs)
        hidden_states = self.patch_embed(pixel_values)
        position_embeddings = self.rotary_pos_emb(hidden_states, position_ids)
        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        hidden_states = self.post_trunk_norm(hidden_states)
        return BaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=self.adapter(hidden_states),
        )


# Unified multimodal model
@auto_docstring
class Dots3NoteModel(Dots3NotePreTrainedModel, InklingModel):
    config_class = Dots3NoteConfig
    input_modalities = ("image", "video", "audio", "text")

    def __init__(self, config: Dots3NoteConfig):
        Dots3NotePreTrainedModel.__init__(self, config)
        self.language_model = Dots3NoteTextModel(config.text_config)
        self.vision_tower = Dots3NoteVisionModel(config.vision_config) if config.vision_config is not None else None
        self.audio_tower = Dots3NoteAudioModel(config.audio_config) if config.audio_config is not None else None
        self.audio_adapter = Dots3NoteAudioAdapter(config.audio_config) if config.audio_config is not None else None
        self.post_init()

    @accepts_precomputed_kwargs(modality="image")
    @can_return_tuple
    @auto_docstring
    def get_image_features(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        **kwargs,
    ) -> BaseModelOutputWithPooling:
        outputs = self.vision_tower(
            pixel_values,
            grid_thw=image_grid_thw,
            return_dict=True,
            **kwargs,
        )

        split_sizes = (image_grid_thw.prod(-1) // self.vision_tower.spatial_merge_size**2).tolist()
        outputs.pooler_output = outputs.pooler_output.split(split_sizes)
        return outputs

    @accepts_precomputed_kwargs(modality="video")
    @can_return_tuple
    @auto_docstring
    def get_video_features(
        self,
        pixel_values_videos: torch.Tensor,
        video_grid_thw: torch.Tensor,
        **kwargs,
    ) -> BaseModelOutputWithPooling:
        return self.get_image_features(pixel_values_videos, video_grid_thw, return_dict=True, **kwargs)

    @can_return_tuple
    @auto_docstring
    def get_audio_features(
        self,
        input_features: torch.Tensor,
        chunk_sample_lengths: torch.Tensor,
        **kwargs,
    ) -> BaseModelOutputWithPooling:
        r"""
        chunk_sample_lengths (`torch.Tensor`):
            Number of waveform samples represented by each feature chunk.
        """
        outputs = self.audio_tower(
            input_features=input_features,
            chunk_sample_lengths=chunk_sample_lengths,
            return_dict=True,
            **kwargs,
        )
        outputs.pooler_output = self.audio_adapter(outputs.pooler_output)
        return outputs

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        input_features: torch.Tensor | None = None,
        chunk_sample_lengths: torch.Tensor | None = None,
        mm_encoder_outputs: dict[str, BaseModelOutputWithPooling] | None = None,
        **kwargs,
    ) -> MoeModelOutputWithPast | tuple:
        r"""
        chunk_sample_lengths (`torch.Tensor`, *optional*):
            Waveform sample count for each audio feature chunk.
        """
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if (pixel_values is not None or pixel_values_videos is not None) and mm_encoder_outputs is not None:
            raise ValueError("Cannot specify both pixel_values/pixel_values_videos and mm_encoder_outputs")

        mm_encoder_outputs = mm_encoder_outputs if mm_encoder_outputs is not None else {}
        if pixel_values is not None:
            mm_encoder_outputs["image"] = self.get_image_features(
                pixel_values, image_grid_thw, return_dict=True, **kwargs
            )
        if mm_encoder_outputs.get("image") is not None:
            image_embeddings = torch.cat(mm_encoder_outputs["image"].pooler_output, dim=0)
            image_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds, image_embeddings, self.config.image_token_id
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeddings.to(inputs_embeds))

        if pixel_values_videos is not None:
            mm_encoder_outputs["video"] = self.get_video_features(
                pixel_values_videos, video_grid_thw, return_dict=True, **kwargs
            )
        if mm_encoder_outputs.get("video") is not None:
            video_embeddings = torch.cat(mm_encoder_outputs["video"].pooler_output, dim=0)
            video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds, video_embeddings, self.config.video_token_id
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeddings.to(inputs_embeds))

        if input_features is not None:
            if chunk_sample_lengths is None:
                raise ValueError("input_features requires chunk_sample_lengths")
            audio_embeddings = self.get_audio_features(
                input_features, chunk_sample_lengths, return_dict=True
            ).pooler_output
            audio_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds, audio_embeddings, self.config.audio_token_id
            )
            inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_embeddings.to(inputs_embeds))

        return self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )


@auto_docstring
class Dots3NoteForConditionalGeneration(Dots3NotePreTrainedModel, VoxtralForConditionalGeneration):
    _keep_in_fp32_modules_strict = AttributeError()
    input_modalities = ("image", "video", "audio", "text")
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}

    def _expand_inputs_for_generation(
        self,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: torch.LongTensor | None = None,
        **model_kwargs,
    ):
        audio_inputs = {}
        if expand_size > 1 and model_kwargs.get("input_features") is not None:
            chunk_sample_lengths = model_kwargs.get("chunk_sample_lengths")
            if chunk_sample_lengths is None:
                raise ValueError("input_features requires chunk_sample_lengths")

            if input_ids is not None and input_ids.numel() > 0:
                audio_mask = input_ids == self.config.audio_token_id
            else:
                inputs_embeds = model_kwargs["inputs_embeds"]
                audio_token = torch.full((), self.config.audio_token_id, dtype=torch.long, device=inputs_embeds.device)
                audio_mask = (inputs_embeds == self.get_input_embeddings()(audio_token)).all(-1)

            stride = self.config.audio_config.samples_per_token
            chunk_tokens = (chunk_sample_lengths + stride - 1) // stride
            chunk_boundaries = F.pad(chunk_tokens.cumsum(0), (1, 0))
            sample_boundaries = F.pad(audio_mask.sum(-1).cumsum(0), (1, 0)).to(chunk_boundaries.device)
            chunk_offsets = torch.searchsorted(chunk_boundaries, sample_boundaries)
            if chunk_offsets[-1] != len(chunk_tokens) or not torch.equal(
                chunk_boundaries[chunk_offsets.clamp(max=len(chunk_tokens))], sample_boundaries
            ):
                raise ValueError("Audio chunks must match the audio placeholder tokens in each sample")

            # Audio chunks are packed across samples; each beam needs its sample's whole chunk group.
            chunk_counts = chunk_offsets.diff().tolist()
            for key in ("input_features", "chunk_sample_lengths"):
                value = model_kwargs.pop(key)
                audio_inputs[key] = torch.cat(
                    [chunk.repeat((expand_size,) + (1,) * (value.ndim - 1)) for chunk in value.split(chunk_counts)]
                )

        input_ids, model_kwargs = super()._expand_inputs_for_generation(
            expand_size=expand_size,
            is_encoder_decoder=is_encoder_decoder,
            input_ids=input_ids,
            **model_kwargs,
        )
        model_kwargs.update(audio_inputs)
        return input_ids, model_kwargs

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        input_features: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        chunk_sample_lengths: torch.Tensor | None = None,
        mm_encoder_outputs: dict[str, BaseModelOutputWithPooling] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeCausalLMOutputWithPast:
        r"""
        chunk_sample_lengths (`torch.Tensor`, *optional*):
            Number of waveform samples in each audio chunk.
        """
        outputs = self.model(
            input_ids=input_ids,
            return_dict=True,
            input_features=input_features,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            chunk_sample_lengths=chunk_sample_lengths,
            mm_encoder_outputs=mm_encoder_outputs,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
            )

        return MoeCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )


@requires(backends=("torch",))
class Dots3NoteFeatureExtractor(WhisperFeatureExtractor):
    """Convert 16 kHz mono waveforms into Dots 3 Note Preview log-mel chunks."""

    model_input_names = ["input_features", "chunk_sample_lengths"]

    def __init__(
        self,
        feature_size: int = 128,
        sampling_rate: int = 16_000,
        padding_value: float = 0.0,
        n_fft: int = 400,
        hop_length: int = 160,
        chunk_length: int = 60,
        chunk_seconds: int | None = None,
        return_attention_mask: bool = False,
        dither: float = 0.0,
        **kwargs,
    ):
        chunk_length = chunk_length if chunk_seconds is None else chunk_seconds
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            n_fft=n_fft,
            hop_length=hop_length,
            chunk_length=chunk_length,
            return_attention_mask=return_attention_mask,
            dither=dither,
            **kwargs,
        )

    def _np_extract_fbank_features(self, *args, **kwargs):
        raise AttributeError("Dots audio feature extraction requires PyTorch")

    def zero_mean_unit_var_norm(self, *args, **kwargs):
        raise AttributeError("Dots audio uses unnormalized waveforms")

    def __call__(
        self,
        raw_speech,
        sampling_rate: int | None = None,
        return_tensors: str | TensorType | None = "pt",
        device: str = "cpu",
        **kwargs,
    ) -> BatchFeature:
        """Extract chunked log-mel features and their waveform sample lengths."""
        if sampling_rate is not None and sampling_rate != self.sampling_rate:
            raise ValueError(f"Expected sampling rate {self.sampling_rate}, got {sampling_rate}")
        if isinstance(raw_speech, (list, tuple)):
            if not raw_speech:
                raise ValueError("received an empty audio batch")
            if isinstance(raw_speech[0], (int, float, np.integer, np.floating)):
                raw_speech = np.asarray(raw_speech)
        chunks = []
        chunk_sample_lengths = []
        chunk_token_lengths = []
        audio_token_lengths = []

        for waveform in make_list_of_audio(raw_speech):
            waveform = torch.as_tensor(waveform, dtype=torch.float32)
            if waveform.ndim == 2 and waveform.shape[0] == 1:
                waveform = waveform.squeeze(0)
            if waveform.ndim != 1:
                raise ValueError(f"Dots 3 Note Preview audio must be mono, got shape={tuple(waveform.shape)}")
            if waveform.numel() == 0:
                raise ValueError("audio waveform must contain at least one sample")
            per_audio_tokens = 0
            for chunk in waveform.split(self.n_samples):
                sample_length = int(chunk.numel())
                # Three stride-2 convolutions downsample the mel frames by a factor of eight.
                token_length = math.ceil(sample_length / (self.hop_length * 8))
                chunks.append(chunk.cpu().numpy()[:, None])
                chunk_sample_lengths.append(sample_length)
                chunk_token_lengths.append(token_length)
                per_audio_tokens += token_length
            audio_token_lengths.append(per_audio_tokens)

        inputs = self.pad(
            BatchFeature({"input_features": chunks}),
            padding="max_length",
            max_length=self.n_samples,
            return_tensors="np",
            return_attention_mask=False,
        )
        input_features = self._torch_extract_fbank_features(inputs["input_features"].squeeze(-1), device)
        input_features = input_features[..., : max(chunk_token_lengths) * 8]
        data = {
            "input_features": torch.from_numpy(input_features).to(device),
            "chunk_sample_lengths": torch.tensor(chunk_sample_lengths, dtype=torch.long, device=device),
            "num_audio_tokens": torch.tensor(audio_token_lengths, dtype=torch.long, device=device),
        }
        return BatchFeature(data=data, tensor_type=return_tensors)


@auto_docstring
class Dots3NoteProcessor(Qwen2VLProcessor):
    valid_processor_kwargs = ProcessingKwargs

    @property
    def unused_input_names(self):
        return ["num_audio_tokens"]

    def replace_audio_token(self, audio_inputs: dict, audio_idx: int, **kwargs) -> str:
        return self.audio_token * audio_inputs["num_audio_tokens"][audio_idx]

    def model_input_names(self):
        raise AttributeError("Use ProcessorMixin.model_input_names")

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        video_processor=None,
        feature_extractor=None,
        chat_template=None,
    ):
        self.image_token = "<|imgpad|>"
        self.video_token = "<|video_pad|>"
        self.audio_token = "<|audio_comp_pad|>"
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        self.video_token_id = tokenizer.convert_tokens_to_ids(self.video_token)
        self.audio_token_id = tokenizer.convert_tokens_to_ids(self.audio_token)
        ProcessorMixin.__init__(
            self,
            image_processor=image_processor,
            tokenizer=tokenizer,
            video_processor=video_processor,
            feature_extractor=feature_extractor,
            chat_template=chat_template,
        )


__all__ = [
    "Dots3NoteAudioConfig",
    "Dots3NoteConfig",
    "Dots3NoteTextConfig",
    "Dots3NoteVisionConfig",
    "Dots3NoteFeatureExtractor",
    "Dots3NoteProcessor",
    "Dots3NoteAudioModel",
    "Dots3NoteAudioPreTrainedModel",
    "Dots3NoteForCausalLM",
    "Dots3NoteForConditionalGeneration",
    "Dots3NoteModel",
    "Dots3NotePreTrainedModel",
    "Dots3NoteTextModel",
    "Dots3NoteVisionModel",
    "Dots3NoteVisionPreTrainedModel",
]
