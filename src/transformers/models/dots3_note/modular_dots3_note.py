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
from torch import nn

from ... import initialization as init
from ...audio_utils import make_list_of_audio
from ...cache_utils import Cache, DynamicCache
from ...feature_extraction_utils import BatchFeature
from ...image_utils import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from ...masking_utils import create_bidirectional_mask, create_causal_mask, create_sliding_window_causal_mask
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPast,
    BaseModelOutputWithPooling,
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
    torch_compilable_check,
)
from ...utils.generic import merge_with_config_defaults
from ...utils.import_utils import requires
from ...utils.output_capturing import capture_outputs
from ...vision_utils import get_vision_attention_seqlens, get_vision_position_ids
from ..deepseek_v3.modeling_deepseek_v3 import DeepseekV3Attention, eager_attention_forward
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
from ..gemma4.modeling_gemma4 import Gemma4Model
from ..glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaIndexer, apply_rotary_pos_emb_interleave
from ..glm_ocr.modeling_glm_ocr import GlmOcrVisionAttention
from ..hy_v3.modeling_hy_v3 import HYV3TopKRouter
from ..llama.modeling_llama import LlamaMLP
from ..mimo_v2_flash.modeling_mimo_v2_flash import MiMoV2FlashMoE
from ..nemotron.modeling_nemotron import NemotronAttention
from ..nemotron_asr_streaming.modeling_nemotron_asr_streaming import _mask_subsampled_frames
from ..phi.modeling_phi import PhiRotaryEmbedding
from ..phi3.modeling_phi3 import Phi3DecoderLayer, Phi3MLP, Phi3PreTrainedModel
from ..qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil
from ..qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessorKwargs
from ..qwen2_vl.modeling_qwen2_vl import (
    PatchMerger,
    Qwen2VisionTransformerPretrainedModel,
    Qwen2VLPreTrainedModel,
    Qwen2VLVisionBlock,
    Qwen2VLVisionRotaryEmbedding,
)
from ..qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor
from ..qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionPatchEmbed
from ..qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor
from ..voxtral.modeling_voxtral import VoxtralForConditionalGeneration
from ..whisper.feature_extraction_whisper import WhisperFeatureExtractor
from .configuration_dots3_note import (
    Dots3NoteAudioConfig,
    Dots3NoteConfig,
    Dots3NoteTextConfig,
    Dots3NoteVisionConfig,
)


logger = logging.get_logger(__name__)


# Text decoder
class Dots3NoteRMSNorm(DeepseekV32RMSNorm):
    pass


class Dots3NoteRotaryEmbedding(DiffusionGemmaTextRotaryEmbedding):
    def __init__(self, config: Dots3NoteTextConfig, device=None):
        super().__init__(config, device)

    @classmethod
    def compute_default_rope_parameters(cls, config: Dots3NoteTextConfig, device=None, layer_type=None, **kwargs):
        return super().compute_default_rope_parameters(config, device, layer_type, **kwargs)


class Dots3NoteTextIndexer(GlmMoeDsaIndexer):
    # TODO: Add FP8 support through the shared indexer and quantization path.
    def __init__(self, config: Dots3NoteTextConfig, layer_idx: int):
        super().__init__(config, layer_idx)


class Dots3NoteTextMLP(DeepseekV32MLP):
    pass


class Dots3NoteTextTopkRouter(DeepseekV32TopkRouter):
    def __init__(self, config: Dots3NoteTextConfig):
        super().__init__(config)


class Dots3NoteTextExperts(DeepseekV32Experts):
    pass


class Dots3NoteTextMoE(DeepseekV32MoE):
    def __init__(self, config):
        super().__init__(config)
        self.shared_experts = Dots3NoteTextMLP(
            config, intermediate_size=config.shared_experts_intermediate_size * config.n_shared_experts
        )


class Dots3NoteTextAttention(DeepseekV3Attention):
    def __init__(self, config: Dots3NoteTextConfig, layer_idx: int):
        layer_config = config.per_layer_config[layer_idx]
        super().__init__(config, layer_idx)
        self.num_heads = layer_config.num_attention_heads
        self.q_lora_rank = layer_config.q_lora_rank
        self.kv_lora_rank = layer_config.kv_lora_rank
        self.qk_rope_head_dim = layer_config.qk_rope_head_dim
        self.qk_nope_head_dim = layer_config.qk_nope_head_dim
        self.v_head_dim = layer_config.v_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.num_key_value_groups = self.num_heads // layer_config.num_key_value_heads
        self.q_a_proj = nn.Linear(self.hidden_size, self.q_lora_rank, bias=config.attention_bias)
        self.q_b_proj = nn.Linear(self.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size, self.kv_lora_rank + self.qk_rope_head_dim, bias=config.attention_bias
        )
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank, self.num_heads * (self.qk_nope_head_dim + self.v_head_dim), bias=False
        )
        self.head_dim = layer_config.head_dim
        self.scaling = self.qk_head_dim**-0.5

        self.sliding_window = (
            # CODEPATH: dots-studio/dots3-note-prev uses a window for SWA, but not DSA.
            layer_config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None
        )

        self.q_a_layernorm = Dots3NoteRMSNorm(self.q_lora_rank, eps=layer_config.rms_norm_eps)
        self.kv_a_layernorm = Dots3NoteRMSNorm(self.kv_lora_rank, eps=layer_config.rms_norm_eps)
        self.k_rope_only_layernorm = Dots3NoteRMSNorm(self.qk_rope_head_dim, eps=layer_config.rms_norm_eps)
        self.q_lora_scale = (self.hidden_size / self.q_lora_rank) ** 0.5
        self.kv_lora_scale = (self.hidden_size / self.kv_lora_rank) ** 0.5
        self.g_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False)

        self.indexer = (
            # CODEPATH: dots-studio/dots3-note-prev uses an indexer for DSA, but not SWA.
            Dots3NoteTextIndexer(layer_config, layer_idx)
            if config.layer_types[layer_idx] == "deepseek_sparse_attention"
            else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
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
        q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)
        q_rot, k_rot = q_rot.to(q_states.dtype), k_rot.to(k_pass.dtype)

        query_states = torch.cat((q_pass, q_rot), dim=-1)
        key_states, value_states = self.expand_kv(k_pass, k_rot)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        if attention_mask is not None and attention_mask.shape[-1] != key_states.shape[-2]:
            attention_mask = attention_mask[..., -key_states.shape[-2] :]

        sparse_indices = None
        if self.indexer is not None:
            if attention_mask.ndim != 4 or attention_mask.shape[1] != 1:
                raise ValueError("DSA requires a shared 4D mask; different per-head masks are not supported")
            topk_indices = self.indexer(
                hidden_states,
                q_resid,
                position_embeddings,
                attention_mask[:, 0],
                kwargs.get("position_ids"),
                past_key_values=past_key_values,
            )
            if self.config._attn_implementation in ("eager", "sdpa"):
                index_mask = (
                    topk_indices.new_ones((batch_size, seq_length, key_states.shape[-2]), dtype=torch.bool)
                    .scatter(-1, topk_indices.long(), False)
                    .unsqueeze(1)
                )
                attention_mask = (
                    attention_mask & ~index_mask
                    if attention_mask.dtype == torch.bool
                    else attention_mask.masked_fill(index_mask, torch.finfo(hidden_states.dtype).min)
                )
            else:
                sparse_indices = topk_indices

        if (
            self.config._attn_implementation == "eager"
            and attention_mask is not None
            and attention_mask.dtype == torch.bool
        ):
            attention_mask = torch.zeros_like(attention_mask, dtype=query_states.dtype).masked_fill(
                ~attention_mask, torch.finfo(query_states.dtype).min
            )
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
    _no_split_modules = ["Dots3NoteTextDecoderLayer"]
    _keep_in_fp32_modules = []
    _keys_to_ignore_on_load_unexpected = [
        r"^model\.(language_model\.)?layers\.46\.",
        r"^model\.(language_model\.)?mtp\.",
    ]

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, Dots3NoteRotaryEmbedding):
            for layer_type in module.layer_types:
                inv_freq, _ = module.compute_default_rope_parameters(
                    module.config.per_layer_config[layer_type], layer_type=layer_type
                )
                init.copy_(getattr(module, f"{layer_type}_inv_freq"), inv_freq)
                init.copy_(getattr(module, f"{layer_type}_original_inv_freq"), inv_freq)


class Dots3NoteTextModel(Dots3NotePreTrainedModel, DeepseekV32Model):
    config_class = Dots3NoteTextConfig

    def __init__(self, config: Dots3NoteTextConfig):
        super().__init__(config)
        self.norm = Dots3NoteRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Dots3NoteRotaryEmbedding(config)

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
    ) -> BaseModelOutputWithPast:
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
                inputs_embeds.float() if layer_type == "deepseek_sparse_attention" else inputs_embeds,
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
                "deepseek_sparse_attention": lambda: create_causal_mask(**mask_kwargs, allow_is_causal_skip=False),
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
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class Dots3NoteForCausalLM(Dots3NotePreTrainedModel, DeepseekV32ForCausalLM):
    config_class = Dots3NoteTextConfig

    def forward(self, **super_kwargs):
        """Run the text decoder and return language-model outputs."""
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


class Dots3NoteAudioEncoderLayer(Phi3DecoderLayer):
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
        frequency_bins = config.feature_size
        for _ in range(3):
            frequency_bins = (frequency_bins + 1) // 2
        self.conv_out = nn.Linear(downsample_size * frequency_bins, hidden_size, bias=False)
        self.hop_length = config.hop_length

    def forward(self, input_features: torch.Tensor, audio_sample_lengths: torch.Tensor) -> torch.Tensor:
        hidden_states = input_features.unsqueeze(1)
        valid_lengths = audio_sample_lengths.to(hidden_states.device) // self.hop_length
        hidden_states = _mask_subsampled_frames(hidden_states.transpose(-1, -2), valid_lengths).transpose(-1, -2)
        for conv in self.convs:
            hidden_states = F.gelu(conv(hidden_states))
            valid_lengths = (valid_lengths + 1) // 2
            hidden_states = _mask_subsampled_frames(hidden_states.transpose(-1, -2), valid_lengths).transpose(-1, -2)
        batch_size, channels, frequency, time = hidden_states.shape
        hidden_states = hidden_states.permute(0, 3, 1, 2).reshape(batch_size, time, channels * frequency)
        return self.conv_out(hidden_states)


class Dots3NoteSpeechEncoder(nn.Module):
    """Bidirectional audio encoder with a convolutional stem and rotary positions."""

    def __init__(self, config: Dots3NoteAudioConfig):
        super().__init__()
        self.config = config
        hidden_size = config.hidden_size
        self.conv_stem = Dots3NoteAudioConvStem(config)
        self.rotary_embedding = Dots3NoteAudioRotaryEmbedding(config)
        self.layers = nn.ModuleList([Dots3NoteAudioEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.layer_norm = Dots3NoteRMSNorm(hidden_size)
        self.dropout = config.dropout

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_features: torch.Tensor,
        audio_sample_lens: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutput:
        """
        audio_sample_lens (`torch.LongTensor` of shape `(num_chunks,)`):
            Number of waveform samples in each audio chunk.
        """
        inputs_embeds = self.conv_stem(input_features, audio_sample_lens)
        stride = self.config.hop_length * 8
        token_lengths = (audio_sample_lens + stride - 1) // stride
        padding_mask = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)[None] < token_lengths[:, None]
        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)[None, :]
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
        return BaseModelOutput(last_hidden_state=self.layer_norm(hidden_states)[padding_mask])


class Dots3NoteAudioAdapter(EvollaFeedForward):
    def __init__(self, dim: int, mult=4):
        super().__init__(dim, mult)
        inner_dim = int(dim * mult)
        self.fc1 = nn.Linear(dim, inner_dim)
        self.fc2 = nn.Linear(inner_dim, inner_dim)


@auto_docstring
class Dots3NoteAudioPreTrainedModel(Phi3PreTrainedModel):
    config: Dots3NoteAudioConfig
    config_class = Dots3NoteAudioConfig
    main_input_name = "input_features"
    _no_split_modules = ["Dots3NoteAudioEncoderLayer"]
    _supports_flex_attn = False
    _can_compile_fullgraph = False
    _version = AttributeError()
    _can_record_outputs = {"hidden_states": Dots3NoteAudioEncoderLayer, "attentions": Dots3NoteAudioAttention}


@auto_docstring
class Dots3NoteAudioModel(Dots3NoteAudioPreTrainedModel):
    def __init__(self, config: Dots3NoteAudioConfig):
        super().__init__(config)
        self.audio_adapter = Dots3NoteAudioAdapter(
            config.adapter_input_size,
            mult=config.adapter_output_size / config.adapter_input_size,
        )
        self.speech_encoder = Dots3NoteSpeechEncoder(config)
        self.post_init()

    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_features: torch.Tensor,
        chunk_sample_lengths: torch.Tensor,
        **kwargs,
    ) -> BaseModelOutput:
        """
        Args:
            chunk_sample_lengths (`torch.Tensor`): Number of waveform samples represented by each feature chunk.
        """
        encoder_output = self.speech_encoder(
            input_features=input_features,
            audio_sample_lens=chunk_sample_lengths,
            return_dict=True,
        ).last_hidden_state

        embeddings = self.audio_adapter(encoder_output)
        return BaseModelOutput(last_hidden_state=embeddings)


# Vision encoder and adapter
class Dots3NoteVisionRotaryEmbedding(Qwen2VLVisionRotaryEmbedding):
    pass


class Dots3NoteVisionPatchEmbed(Qwen3VLVisionPatchEmbed):
    def __init__(self, config: Dots3NoteVisionConfig):
        super().__init__(config)
        del self.temporal_patch_size
        self.in_channels = config.num_channels
        kernel_size = self.patch_size
        self.proj = nn.Conv2d(self.in_channels, self.embed_dim, kernel_size=kernel_size, stride=kernel_size)
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


class Dots3NoteVisionTopkRouter(HYV3TopKRouter):
    def __init__(self, config: Dots3NoteVisionConfig):
        super().__init__(config)
        self.e_score_correction_bias = nn.Buffer(torch.zeros(self.num_experts, dtype=torch.float32))

    def forward(self, hidden_states):
        e_score_correction_bias = self.e_score_correction_bias
        return super().forward(hidden_states, e_score_correction_bias)


class Dots3NoteVisionMoE(MiMoV2FlashMoE):
    """Routed vision experts without shared experts."""

    def __init__(self, config: Dots3NoteVisionConfig):
        nn.Module.__init__(self)
        self.experts = Dots3NoteTextExperts(config)
        self.gate = Dots3NoteVisionTopkRouter(config)


class Dots3NoteVisionAttention(GlmOcrVisionAttention):
    def __init__(self, config: Dots3NoteVisionConfig):
        super().__init__(config)
        self.num_heads = config.num_attention_heads
        self.is_causal = False
        self.q_norm = Dots3NoteRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Dots3NoteRMSNorm(self.head_dim, eps=config.rms_norm_eps)


class Dots3NoteVisionBlock(Qwen2VLVisionBlock):
    def __init__(self, config: Dots3NoteVisionConfig, layer_idx: int):
        nn.Module.__init__(self)
        self.norm1 = Dots3NoteRMSNorm(config.embed_dim, eps=config.rms_norm_eps)
        self.attn = Dots3NoteVisionAttention(config)
        self.norm2 = Dots3NoteRMSNorm(config.embed_dim, eps=config.rms_norm_eps)
        layer_config = config.per_layer_config[layer_idx]
        self.mlp = (
            # CODEPATH: dots-studio/dots3-note-prev has both sparse and dense vision MLPs.
            Dots3NoteVisionMoE(layer_config)
            if config.mlp_layer_types[layer_idx] == "sparse"
            else Dots3NoteVisionMLP(layer_config)
        )


class Dots3NoteVisionAdapter(PatchMerger):
    pass


@auto_docstring
class Dots3NoteVisionPreTrainedModel(Qwen2VLPreTrainedModel):
    config: Dots3NoteVisionConfig
    config_class = Dots3NoteVisionConfig
    main_input_name = "pixel_values"
    input_modalities = ("image", "video")
    _no_split_modules = ["Dots3NoteVisionBlock"]
    _can_compile_fullgraph = False

    @torch.no_grad()
    def _init_weights(self, module: nn.Module):
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, Dots3NoteVisionTopkRouter):
            init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            init.zeros_(module.e_score_correction_bias)
        elif isinstance(module, Dots3NoteTextExperts):
            init.normal_(module.gate_up_proj, mean=0.0, std=self.config.initializer_range)
            init.normal_(module.down_proj, mean=0.0, std=self.config.initializer_range)


@auto_docstring
class Dots3NoteVisionModel(Dots3NoteVisionPreTrainedModel, Qwen2VisionTransformerPretrainedModel):
    _can_record_outputs = {"hidden_states": Dots3NoteVisionBlock}

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
        self.post_init()

    def get_dtype(self):
        raise AttributeError("Use the standard dtype property")

    def get_device(self):
        raise AttributeError("Use the standard device property")

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor, **kwargs) -> BaseModelOutputWithPooling:
        """
        Args:
            grid_thw (`torch.Tensor`): Temporal, height, and width patch-grid dimensions for each input.
        """
        kwargs.pop("is_causal", None)
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
class Dots3NoteModel(Dots3NotePreTrainedModel, Gemma4Model):
    accepts_loss_kwargs = AttributeError()
    config_class = Dots3NoteConfig
    input_modalities = ("image", "video", "audio", "text")
    _no_split_modules = [
        "Dots3NoteAudioEncoderLayer",
        "Dots3NoteTextDecoderLayer",
        "Dots3NoteVisionBlock",
    ]

    def __init__(self, config: Dots3NoteConfig):
        Dots3NotePreTrainedModel.__init__(self, config)
        self.language_model = Dots3NoteTextModel(config.text_config)
        self.vision_encoder = Dots3NoteVisionModel(config.vision_config)
        self.audio_encoder = Dots3NoteAudioModel(config.audio_config)
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def get_image_features(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        **kwargs,
    ) -> BaseModelOutputWithPooling:
        """Encode image patches and return the vision model output."""
        return self.vision_encoder(
            pixel_values,
            grid_thw=image_grid_thw,
            return_dict=True,
            **kwargs,
        )

    @can_return_tuple
    @auto_docstring
    def get_video_features(
        self,
        pixel_values_videos: torch.Tensor,
        video_grid_thw: torch.Tensor,
        **kwargs,
    ) -> BaseModelOutputWithPooling:
        """Encode video patches with the shared vision encoder."""
        return self.get_image_features(pixel_values_videos, video_grid_thw, return_dict=True, **kwargs)

    @can_return_tuple
    @auto_docstring
    def get_audio_features(
        self,
        input_features: torch.Tensor,
        chunk_sample_lengths: torch.Tensor,
    ) -> BaseModelOutput:
        """
        Args:
            chunk_sample_lengths (`torch.Tensor`): Number of waveform samples represented by each feature chunk.
        """
        parameter = next(self.audio_encoder.parameters())
        return self.audio_encoder(
            input_features=input_features.to(device=parameter.device, dtype=parameter.dtype),
            chunk_sample_lengths=chunk_sample_lengths.to(parameter.device),
            return_dict=True,
        )

    def get_per_layer_input_embeddings(self):
        raise AttributeError("Dots has no per-layer input embeddings")

    def set_per_layer_input_embeddings(self, value):  # trf-ignore: TRF033 -- modular deletes this inherited method
        raise AttributeError("Dots has no per-layer input embeddings")

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
        **kwargs,
    ) -> BaseModelOutputWithPast | tuple:
        """
        Args:
            chunk_sample_lengths (`torch.Tensor`, *optional*): Waveform sample count for each audio feature chunk.
        """
        has_multimodal_inputs = any(value is not None for value in (pixel_values, pixel_values_videos, input_features))
        if has_multimodal_inputs:
            if (input_ids is None) == (inputs_embeds is None):
                raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
            if inputs_embeds is None:
                inputs_embeds = self.get_input_embeddings()(input_ids)
            image_mask, video_mask, audio_mask = self.get_placeholder_mask(input_ids, inputs_embeds)

            if pixel_values is not None:
                if image_grid_thw is None:
                    raise ValueError("image_grid_thw is required when pixel_values is provided")
                image_embeddings = self.get_image_features(
                    pixel_values, image_grid_thw, return_dict=True
                ).pooler_output
                torch_compilable_check(
                    image_mask.sum() * inputs_embeds.shape[-1] == image_embeddings.numel(),
                    "image embedding/token mismatch: placeholder tokens must match encoder features",
                )
                inputs_embeds = inputs_embeds.masked_scatter(
                    image_mask.unsqueeze(-1), image_embeddings.to(inputs_embeds)
                )

            if pixel_values_videos is not None:
                if video_grid_thw is None:
                    raise ValueError("video_grid_thw is required when pixel_values_videos is provided")
                video_embeddings = self.get_video_features(
                    pixel_values_videos, video_grid_thw, return_dict=True
                ).pooler_output
                torch_compilable_check(
                    video_mask.sum() * inputs_embeds.shape[-1] == video_embeddings.numel(),
                    "video embedding/token mismatch: placeholder tokens must match encoder features",
                )
                inputs_embeds = inputs_embeds.masked_scatter(
                    video_mask.unsqueeze(-1), video_embeddings.to(inputs_embeds)
                )

            if input_features is not None:
                if chunk_sample_lengths is None:
                    raise ValueError("input_features requires chunk_sample_lengths")
                audio_embeddings = self.get_audio_features(
                    input_features, chunk_sample_lengths, return_dict=True
                ).last_hidden_state
                torch_compilable_check(
                    audio_mask.sum() * inputs_embeds.shape[-1] == audio_embeddings.numel(),
                    "audio embedding/token mismatch: placeholder tokens must match encoder features",
                )
                inputs_embeds = inputs_embeds.masked_scatter(
                    audio_mask.unsqueeze(-1), audio_embeddings.to(inputs_embeds)
                )
            input_ids = None

        return self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )


@auto_docstring
class Dots3NoteForConditionalGeneration(Dots3NotePreTrainedModel, VoxtralForConditionalGeneration):
    _keep_in_fp32_modules_strict = []
    input_modalities = ("image", "video", "audio", "text")
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}
    _no_split_modules = ["Dots3NoteAudioEncoderLayer", "Dots3NoteTextDecoderLayer", "Dots3NoteVisionBlock"]

    def __init__(self, config: Dots3NoteConfig):
        Dots3NotePreTrainedModel.__init__(self, config)
        self.model = Dots3NoteModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

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
        **kwargs: Unpack[TransformersKwargs],
    ):
        """
        chunk_sample_lengths (`torch.Tensor`, *optional*):
            Number of waveform samples in each audio chunk.
        """
        kwargs.update(
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            chunk_sample_lengths=chunk_sample_lengths,
        )
        return super().forward(
            input_ids=input_ids,
            input_features=input_features,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            logits_to_keep=logits_to_keep,
            **kwargs,
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

    def _get_num_multimodal_tokens(self, *args, **kwargs):
        raise AttributeError("Use ProcessorMixin multimodal token counting")

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        video_processor=None,
        feature_extractor=None,
        chat_template=None,
    ):
        self.image_token = "<|imgpad|>"
        self.image_start_token = "<|img|>"
        self.image_end_token = "<|endofimg|>"
        self.video_token = "<|video_pad|>"
        self.audio_token = "<|audio_comp_pad|>"
        self.audio_start_token = "<|audio_comp_start|>"
        self.audio_end_token = "<|audio_comp_end|>"
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        self.video_token_id = tokenizer.convert_tokens_to_ids(self.video_token)
        self.audio_token_id = tokenizer.convert_tokens_to_ids(self.audio_token)
        self.audio_start_token_id = tokenizer.convert_tokens_to_ids(self.audio_start_token)
        self.audio_end_token_id = tokenizer.convert_tokens_to_ids(self.audio_end_token)
        ProcessorMixin.__init__(
            self,
            image_processor=image_processor,
            tokenizer=tokenizer,
            video_processor=video_processor,
            feature_extractor=feature_extractor,
            chat_template=chat_template,
        )


class Dots3NoteImageProcessorKwargs(Qwen2VLImageProcessorKwargs):
    pass


@auto_docstring
class Dots3NoteImageProcessorPil(Qwen2VLImageProcessorPil):
    """Qwen2-VL PIL preprocessing with Dots image defaults."""

    size = {"shortest_edge": 56 * 56, "longest_edge": (36 * 28) ** 2}
    temporal_patch_size = 1


class Dots3NoteVideoProcessor(Qwen3VLVideoProcessor):
    size = {"shortest_edge": 56 * 56, "longest_edge": (36 * 28) ** 2}
    image_mean = OPENAI_CLIP_MEAN
    image_std = OPENAI_CLIP_STD
    patch_size = 14
    temporal_patch_size = 1
    fps = 1


__all__ = [
    "Dots3NoteFeatureExtractor",
    "Dots3NoteProcessor",
    "Dots3NoteImageProcessorPil",
    "Dots3NoteVideoProcessor",
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
