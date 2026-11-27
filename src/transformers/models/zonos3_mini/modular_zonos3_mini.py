# Copyright 2026 Zyphra and The HuggingFace Inc. team. All rights reserved.
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
#
# Ported from the ZONOS3-Mini release (https://huggingface.co/Zyphra/ZONOS3-Mini-hf, Apache-2.0, Copyright Zyphra). The
# speaker encoder is the MECT-B2 backbone of AntSpeaker (Apache-2.0, Copyright (c) 2026 Ant Group Co., Ltd.).
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict

from ... import initialization as init
from ...activations import ACT2FN
from ...audio_utils import AudioInput, make_list_of_audio
from ...cache_utils import Cache, DynamicCache
from ...configuration_utils import PreTrainedConfig, SubConfigSpec
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...integrations import use_experts_implementation
from ...masking_utils import create_causal_mask, create_recurrent_attention_mask
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import ModelOutput, MoeModelOutputWithPast
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TensorType, TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.generic import merge_with_config_defaults
from ...utils.import_utils import is_torchaudio_available, requires
from ...utils.output_capturing import OutputRecorder, capture_outputs
from ..csm.modeling_csm import CsmBackboneModelEmbeddings
from ..deepseek_v3.modeling_deepseek_v3 import apply_rotary_pos_emb_interleave
from ..deepseek_v4.modeling_deepseek_v4 import DeepseekV4UnweightedRMSNorm
from ..llama.modeling_llama import LlamaRMSNorm, eager_attention_forward
from ..olmo_hybrid.modeling_olmo_hybrid import OlmoHybridRotaryEmbedding
from ..qwen2_moe.modeling_qwen2_moe import Qwen2MoeExperts, Qwen2MoeMLP
from ..qwen3.modeling_qwen3 import Qwen3Attention
from ..qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet
from ..qwen3_next.configuration_qwen3_next import Qwen3NextConfig
from ..resnet.modeling_resnet import ResNetShortCut
from ..speech_to_text.feature_extraction_speech_to_text import Speech2TextFeatureExtractor
from ..wav2vec2.modeling_wav2vec2 import Wav2Vec2Attention, Wav2Vec2EncoderLayer, Wav2Vec2FeedForward
from .generation_zonos3_mini import Zonos3MiniGenerationMixin


if is_torchaudio_available():
    import torchaudio


# The MECT-B2 stages of the release: per block `[num_res_blocks, out_channels, freq_stride, time_stride, expansion]`
SPEAKER_ENCODER_STAGES = [
    [[3, 32, 1, 1, 4], [3, 32, 1, 1, 4]],
    [[3, 64, 2, 1, 2], [3, 64, 1, 2, 2], [3, 64, 1, 2, 2], [3, 64, 1, 2, 2]],
    [[3, 128, 2, 1, 2], [3, 128, 1, 2, 2], [3, 128, 1, 2, 2], [3, 128, 1, 2, 2], [3, 128, 1, 2, 2]],
    [[3, 256, 2, 1, 1], [2, 256, 1, 2, 1], [2, 256, 1, 2, 1]],
]


@auto_docstring(checkpoint="Zyphra/ZONOS3-Mini-transformers")
@strict
class Zonos3MiniSpeakerEncoderConfig(PreTrainedConfig):
    r"""
    num_mel_bins (`int`, *optional*, defaults to 80):
        Number of Kaldi mel filterbank bins of the input features.
    in_channels (`int`, *optional*, defaults to 32):
        Number of channels of the convolutional stem.
    num_experts (`int`, *optional*, defaults to 4):
        Number of experts of the soft mixture-of-experts feed-forward of each transformer layer.
    bottleneck_size (`int`, *optional*, defaults to 128):
        Hidden size of the attentive statistics pooling.
    embedding_size (`int`, *optional*, defaults to 192):
        Size of the speaker embedding.
    pos_conv_kernel_size (`int`, *optional*, defaults to 113):
        Kernel size of the depthwise convolutional positional embedding of each block.
    channel_norm_eps (`float`, *optional*, defaults to 1e-06):
        Epsilon of the channel layer norm of each block.
    batch_norm_eps (`float`, *optional*, defaults to 1e-05):
        Epsilon of the batch norms.
    activation_dropout (`float`, *optional*, defaults to 0.0):
        Dropout probability after the activation of the feed-forward layers.
    stage_block_specs (`list[list[list[int]]]`, *optional*):
        The blocks of each stage, as `[num_res_blocks, out_channels, freq_stride, time_stride, expansion]`. Defaults
        to the release's MECT-B2 stages.

    Example:

    ```python
    >>> from transformers import Zonos3MiniSpeakerEncoderConfig, Zonos3MiniSpeakerEncoderModel

    >>> configuration = Zonos3MiniSpeakerEncoderConfig()
    >>> model = Zonos3MiniSpeakerEncoderModel(configuration)
    ```"""

    model_type = "zonos3_mini_speaker_encoder"
    base_config_key = "speaker_encoder_config"

    num_mel_bins: int = 80
    in_channels: int = 32
    hidden_size: int = 64
    num_attention_heads: int = 4
    num_experts: int = 4
    bottleneck_size: int = 128
    embedding_size: int = 192
    pos_conv_kernel_size: int = 113
    layer_norm_eps: float = 1e-6
    channel_norm_eps: float = 1e-6
    batch_norm_eps: float = 1e-5
    hidden_act: str = "gelu_new"
    attention_dropout: float | int = 0.0
    hidden_dropout: float | int = 0.0
    activation_dropout: float | int = 0.0
    stage_block_specs: list | None = None
    initializer_range: float = 0.02

    def __post_init__(self, **kwargs):
        if self.stage_block_specs is None:
            self.stage_block_specs = [[list(block) for block in stage] for stage in SPEAKER_ENCODER_STAGES]
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        # every block keeps `channels * frequency bins` equal to `in_channels * num_mel_bins` (the stage outputs are
        # stacked), so a block's output channels are its input channels times its frequency stride
        channels = self.in_channels
        for stage in self.stage_block_specs:
            for _, out_channels, freq_stride, _, _ in stage:
                if out_channels != channels * freq_stride:
                    raise ValueError(
                        f"Each block's out_channels must be its input channels times its freq_stride: got "
                        f"{out_channels} for {channels} input channels and freq_stride {freq_stride}."
                    )
                channels = out_channels


@auto_docstring(checkpoint="Zyphra/ZONOS3-Mini-transformers")
@strict
class Zonos3MiniConfig(Qwen3NextConfig):
    r"""
    linear_conv_kernel_dim (`int`, *optional*, defaults to 4):
        Kernel size of the causal convolution of the gated DeltaNet layers.
    linear_key_head_dim (`int`, *optional*, defaults to 128):
        Dimension of each key head of the gated DeltaNet layers.
    linear_value_head_dim (`int`, *optional*, defaults to 128):
        Dimension of each value head of the gated DeltaNet layers.
    linear_num_key_heads (`int`, *optional*, defaults to 4):
        Number of key heads of the gated DeltaNet layers.
    linear_num_value_heads (`int`, *optional*, defaults to 4):
        Number of value heads of the gated DeltaNet layers.
    qk_norm_eps (`float`, *optional*, defaults to 1e-06):
        Epsilon of the (unweighted) query and key norms of the attention layers.
    moe_latent_size (`int`, *optional*, defaults to 256):
        Size of the latent space the routed experts operate in: tokens are projected down to it before the experts
        and back up after.
    mlp_layer_types (`list[str]`, *optional*):
        The feed-forward of each layer: `"dense"` or `"sparse"`. Defaults to a dense first layer and sparse others.
    num_codebooks (`int`, *optional*, defaults to 10):
        Number of audio codebooks of each row.
    codebook_size (`int`, *optional*, defaults to 4096):
        Number of codes of each codebook. The audio vocabulary adds the end-of-audio and padding ids.
    audio_eos_token_id (`int`, *optional*, defaults to 4096):
        The end-of-audio id of the audio columns.
    audio_pad_token_id (`int`, *optional*, defaults to 4097):
        The padding id of the audio columns.
    text_pad_token_id (`int`, *optional*, defaults to 8196):
        The padding id of the text column.
    speaker_embedding_size (`int`, *optional*, defaults to 192):
        Size of the speaker embeddings projected into the speaker row.
    condition_bucket_counts (`list[int]`, *optional*, defaults to `[10, 10, 30]`):
        Number of buckets of each condition (loudness, fmax, speaking rate); bucket `count` means unknown.
    final_logit_softcapping (`float`, *optional*, defaults to 15.0):
        Softcap of the logits: `cap * tanh(logits / cap)`.
    speaker_encoder_config (`Union[dict, Zonos3MiniSpeakerEncoderConfig]`, *optional*):
        Configuration of the speaker encoder.

    Example:

    ```python
    >>> from transformers import Zonos3MiniConfig, Zonos3MiniForConditionalGeneration

    >>> configuration = Zonos3MiniConfig()
    >>> model = Zonos3MiniForConditionalGeneration(configuration)
    ```"""

    model_type = "zonos3_mini"
    sub_configs_defaults = {"speaker_encoder_config": SubConfigSpec(config_class=Zonos3MiniSpeakerEncoderConfig)}
    base_model_tp_plan = AttributeError()
    base_model_pp_plan = AttributeError()
    base_model_ep_plan = AttributeError()

    vocab_size: int = 8197
    hidden_size: int = 1024
    intermediate_size: int = 2048
    num_hidden_layers: int = 11
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 32
    rms_norm_eps: float = 1e-5
    qk_norm_eps: float = 1e-6
    linear_num_key_heads: int = 4
    linear_num_value_heads: int = 4
    moe_intermediate_size: int = 1536
    moe_latent_size: int = 256
    num_experts: int = 64
    num_experts_per_tok: int = 2
    mlp_layer_types: list[str] | None = None
    num_codebooks: int = 10
    codebook_size: int = 4096
    audio_eos_token_id: int = 4096
    audio_pad_token_id: int = 4097
    text_pad_token_id: int = 8196
    speaker_embedding_size: int = 192
    condition_bucket_counts: list[int] | tuple[int, ...] = (10, 10, 30)
    final_logit_softcapping: float = 15.0
    speaker_encoder_config: dict | PreTrainedConfig | None = None
    pad_token_id: int | None = 0
    bos_token_id: int | None = 2
    eos_token_id: int | list[int] | None = 3

    decoder_sparse_step = AttributeError()
    norm_topk_prob = AttributeError()
    mlp_only_layers = AttributeError()
    shared_expert_intermediate_size = AttributeError()
    output_router_logits = AttributeError()
    router_aux_loss_coef = AttributeError()

    def __post_init__(self, **kwargs):
        if self.rope_parameters is None:
            self.rope_parameters = {"rope_type": "default", "rope_theta": 10000.0}
        if self.layer_types is None:
            self.layer_types = [
                "linear_attention" if (i + 1) % 4 else "full_attention" for i in range(self.num_hidden_layers)
            ]
        if self.mlp_layer_types is None:
            self.mlp_layer_types = ["dense"] + ["sparse"] * (self.num_hidden_layers - 1)
        PreTrainedConfig.__post_init__(self, **kwargs)

    @property
    def audio_vocab_size(self) -> int:
        """Size of each audio column's vocabulary: the codes, end-of-audio and padding."""
        return self.codebook_size + 2


@auto_docstring
@dataclass
class Zonos3MiniSpeakerEncoderOutput(ModelOutput):
    r"""
    speaker_embeddings (`torch.FloatTensor` of shape `(batch_size, embedding_size)`):
        The (unnormalized) speaker embeddings, in float32.
    """

    speaker_embeddings: torch.FloatTensor | None = None


@auto_docstring
@dataclass
class Zonos3MiniOutputWithPast(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Next-row cross-entropy of the audio columns.
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_codebooks, codebook_size + 2)`):
        Softcapped logits of each audio column.
    cache (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        The cache of the attention and gated DeltaNet layers.
    router_logits (`tuple(torch.FloatTensor)`, *optional*, returned when `output_router_logits=True`):
        Router logits of the sparse layers.
    """

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    cache: Cache | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None
    router_logits: tuple[torch.FloatTensor, ...] | None = None


class Zonos3MiniSpeakerEncoderAttention(Wav2Vec2Attention):
    pass


class Zonos3MiniSpeakerEncoderSoftMoE(nn.Module):
    """Soft mixture of experts: every token goes through every expert, weighted by a softmax gate."""

    def __init__(self, config: Zonos3MiniSpeakerEncoderConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.Linear(config.hidden_size, config.hidden_size * config.num_experts, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        weights = F.softmax(self.gate(hidden_states), dim=-1, dtype=torch.float32)
        expert_states = self.act_fn(self.experts(hidden_states))
        expert_states = expert_states.view(*hidden_states.shape[:-1], self.num_experts, hidden_states.shape[-1])
        return (expert_states * weights[..., None]).sum(-2).to(hidden_states.dtype)


class Zonos3MiniSpeakerEncoderFeedForward(Wav2Vec2FeedForward):
    def __init__(self, config: Zonos3MiniSpeakerEncoderConfig):
        nn.Module.__init__(self)
        self.intermediate_dropout = nn.Dropout(config.activation_dropout)
        self.intermediate_dense = Zonos3MiniSpeakerEncoderSoftMoE(config)
        self.intermediate_act_fn = ACT2FN[config.hidden_act]
        self.output_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.output_dropout = nn.Dropout(config.hidden_dropout)


class Zonos3MiniSpeakerEncoderEncoderLayer(Wav2Vec2EncoderLayer):
    pass


class Zonos3MiniSpeakerEncoderShortCut(ResNetShortCut):
    def __init__(self, config: Zonos3MiniSpeakerEncoderConfig, in_channels: int, out_channels: int, stride):
        super().__init__(in_channels, out_channels, stride=stride)
        self.normalization = nn.BatchNorm2d(out_channels, eps=config.batch_norm_eps)


class Zonos3MiniSpeakerEncoderResidualBlock(nn.Module):
    """Depthwise-separable 3x3 residual block (AntSpeaker `BasicResBlock`)."""

    def __init__(self, config: Zonos3MiniSpeakerEncoderConfig, in_channels: int, out_channels: int, stride):
        super().__init__()
        eps = config.batch_norm_eps
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, groups=in_channels, bias=False)
        self.conv_pw1 = nn.Conv2d(out_channels, out_channels, 1)
        self.bn1 = nn.BatchNorm2d(out_channels, eps=eps)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=in_channels, bias=False)
        self.conv_pw2 = nn.Conv2d(out_channels, out_channels, 1)
        self.bn2 = nn.BatchNorm2d(out_channels, eps=eps)
        if tuple(stride) != (1, 1) or in_channels != out_channels:
            self.shortcut = Zonos3MiniSpeakerEncoderShortCut(config, in_channels, out_channels, stride)
        else:
            self.shortcut = nn.Identity()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(hidden_states)
        hidden_states = F.relu(self.bn1(self.conv_pw1(self.conv1(hidden_states))))
        hidden_states = F.relu(self.bn2(self.conv_pw2(self.conv2(hidden_states))))
        return hidden_states + residual


class Zonos3MiniSpeakerEncoderPositionalConv(nn.Module):
    def __init__(self, config: Zonos3MiniSpeakerEncoderConfig):
        super().__init__()
        size, kernel_size = config.hidden_size, config.pos_conv_kernel_size
        self.conv = nn.Conv1d(size, size, kernel_size, padding=kernel_size // 2, groups=size)
        self.norm = nn.BatchNorm1d(size, eps=config.batch_norm_eps)
        self.act = nn.GELU()
        self.proj = nn.Conv1d(size, size, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.proj(self.act(self.norm(self.conv(hidden_states))))


class Zonos3MiniSpeakerEncoderMECTBlock(nn.Module):
    """
    MECT block: residual conv blocks on `[batch, channels, freq, time]`, then a transformer layer on the flattened
    `channels x freq` features (time-strided blocks run it at a lower frame rate and repeat its output back).
    """

    def __init__(
        self,
        config: Zonos3MiniSpeakerEncoderConfig,
        num_res_blocks: int,
        in_channels: int,
        out_channels: int,
        num_freq_bins: int,
        stride: tuple[int, int],
        expansion: int,
    ):
        super().__init__()
        self.freq_stride, self.time_stride = stride
        channels = in_channels
        res_blocks = []
        for i in range(num_res_blocks):
            res_blocks.append(
                Zonos3MiniSpeakerEncoderResidualBlock(
                    config, channels, out_channels * expansion, stride if i == 0 else (1, 1)
                )
            )
            channels = out_channels * expansion
        self.res_blocks = nn.ModuleList(res_blocks)
        if expansion > 1:
            self.reduce_conv = nn.Conv2d(channels, in_channels * self.freq_stride, 1)
            self.reduce_norm = nn.BatchNorm2d(in_channels * self.freq_stride, eps=config.batch_norm_eps)
        else:
            self.reduce_conv = self.reduce_norm = None
        flat_size = num_freq_bins * out_channels // self.freq_stride
        self.reduce_proj = nn.Conv1d(flat_size, config.hidden_size, 1)
        self.reduce_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.channel_norm_eps)
        self.pos_conv = Zonos3MiniSpeakerEncoderPositionalConv(config)
        self.transformer = Zonos3MiniSpeakerEncoderEncoderLayer(config)
        self.expansion_proj = nn.Conv1d(config.hidden_size, flat_size, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, _, _, num_frames = hidden_states.shape
        residual = hidden_states
        for block in self.res_blocks:
            hidden_states = block(hidden_states)
        if self.reduce_conv is not None:
            hidden_states = self.reduce_norm(self.reduce_conv(hidden_states))
        _, channels, freq_bins, _ = hidden_states.shape
        flat = hidden_states.permute(0, 2, 1, 3).reshape(batch_size, channels * freq_bins, -1)

        hidden_states = self.reduce_layer_norm(self.reduce_proj(flat).transpose(1, 2)).transpose(1, 2)
        hidden_states = hidden_states + self.pos_conv(hidden_states)
        hidden_states = self.transformer(hidden_states.transpose(1, 2)).transpose(1, 2)
        hidden_states = self.expansion_proj(hidden_states) + flat

        hidden_states = hidden_states.repeat_interleave(self.time_stride, -1)[..., :num_frames]
        hidden_states = hidden_states + residual.permute(0, 2, 1, 3).reshape(batch_size, channels * freq_bins, -1)
        return hidden_states.reshape(batch_size, freq_bins, channels, num_frames).permute(0, 2, 1, 3)


class Zonos3MiniSpeakerEncoderStatisticsPooling(nn.Module):
    """Attentive statistics pooling: attention-weighted mean and standard deviation over time."""

    def __init__(self, config: Zonos3MiniSpeakerEncoderConfig):
        super().__init__()
        size = config.in_channels * config.num_mel_bins
        self.linear1 = nn.Conv1d(size, config.bottleneck_size, 1)
        self.linear2 = nn.Conv1d(config.bottleneck_size, size, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.linear2(torch.tanh(self.linear1(hidden_states))), dim=2)
        mean = torch.sum(weights * hidden_states, dim=2)
        variance = torch.sum(weights * hidden_states**2, dim=2) - mean**2
        return torch.cat([mean, torch.sqrt(variance.clamp(min=1e-7))], dim=1)


class Zonos3MiniRMSNorm(LlamaRMSNorm):
    pass


class Zonos3MiniUnweightedRMSNorm(DeepseekV4UnweightedRMSNorm):
    pass


class Zonos3MiniRotaryEmbedding(OlmoHybridRotaryEmbedding):
    pass


class Zonos3MiniGatedDeltaNet(Qwen3_5GatedDeltaNet):
    pass


class Zonos3MiniAttention(Qwen3Attention):
    """Gated attention with unweighted query/key norms, a learned per-head temperature and interleaved RoPE."""

    def __init__(self, config: Zonos3MiniConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.q_norm = Zonos3MiniUnweightedRMSNorm(config.qk_norm_eps)
        self.k_norm = Zonos3MiniUnweightedRMSNorm(config.qk_norm_eps)
        self.g_proj = nn.Linear(config.hidden_size, config.num_attention_heads, bias=False)
        self.temperature = nn.Parameter(torch.ones(config.num_attention_heads))

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        cache: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        query_states = query_states * self.temperature.abs().view(1, -1, 1, 1)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # float32 cos/sin: the rotation runs in float32, as the original does
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_interleave(query_states, key_states, cos, sin)
        query_states, key_states = query_states.to(value_states.dtype), key_states.to(value_states.dtype)

        if cache is not None:
            key_states, value_states = cache.update(key_states, value_states, self.layer_idx, kwargs=kwargs)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
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
            **kwargs,
        )
        attn_output = attn_output * torch.sigmoid(self.g_proj(hidden_states)).unsqueeze(-1)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Zonos3MiniMLP(Qwen2MoeMLP):
    pass


@use_experts_implementation
class Zonos3MiniExperts(Qwen2MoeExperts):
    def __init__(self, config: Zonos3MiniConfig):
        nn.Module.__init__(self)
        self.num_experts = config.num_experts
        self.hidden_dim = config.moe_latent_size
        self.intermediate_dim = config.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
        self.act_fn = ACT2FN[config.hidden_act]


class Zonos3MiniTopKRouter(nn.Module):
    """Top-k softmax router on RMS-normalized tokens, with expert-balancing biases used for the selection only."""

    def __init__(self, config: Zonos3MiniConfig):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.norm = Zonos3MiniRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.weight = nn.Parameter(torch.zeros(config.num_experts, config.hidden_size))
        self.balancing_biases = nn.Buffer(torch.zeros(config.num_experts))

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        router_logits = F.linear(self.norm(hidden_states), self.weight)
        top_k_index = torch.topk(router_logits.float() - self.balancing_biases.float(), self.top_k, dim=-1).indices
        top_k_weights = router_logits.float().gather(-1, top_k_index).softmax(-1)
        return router_logits, top_k_weights, top_k_index


class Zonos3MiniSparseMoeBlock(nn.Module):
    """NemotronH latent MoE without shared experts: the routed experts run in a lower-dimensional latent space."""

    def __init__(self, config: Zonos3MiniConfig):
        super().__init__()
        self.gate = Zonos3MiniTopKRouter(config)
        self.fc1_latent_proj = nn.Linear(config.hidden_size, config.moe_latent_size, bias=False)
        self.experts = Zonos3MiniExperts(config)
        self.fc2_latent_proj = nn.Linear(config.moe_latent_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_size)
        _, top_k_weights, top_k_index = self.gate(hidden_states)
        hidden_states = self.fc1_latent_proj(hidden_states)
        hidden_states = self.experts(hidden_states, top_k_index, top_k_weights)
        hidden_states = self.fc2_latent_proj(hidden_states)
        return hidden_states.view(batch_size, sequence_length, hidden_size)


class Zonos3MiniDepthRouter(nn.Module):
    """
    Full attention residuals: each sublayer reads a softmax mix, over depth, of the embeddings and every earlier
    sublayer output, scored against a learned query. The closest library module is `Inkling2BlockAttentionResidual`;
    here the mixed values are the un-normalized sources and the scores are not scaled.
    """

    def __init__(self, config: Zonos3MiniConfig):
        super().__init__()
        self.norm = Zonos3MiniUnweightedRMSNorm(config.rms_norm_eps)

    def forward(self, sources: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        """`sources` `[num_sources, batch, seq, hidden]`, `query` `[hidden]` -> `[batch, seq, hidden]`."""
        scores = (self.norm(sources) * query).sum(-1, dtype=torch.float32)
        weights = scores.softmax(0).to(sources.dtype)
        return (sources * weights.unsqueeze(-1)).sum(0)


class Zonos3MiniDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Zonos3MiniConfig, layer_idx: int):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]
        self.input_layernorm = Zonos3MiniRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Zonos3MiniRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if self.layer_type == "linear_attention":
            self.linear_attn = Zonos3MiniGatedDeltaNet(config, layer_idx)
        else:
            self.self_attn = Zonos3MiniAttention(config, layer_idx)
        if config.mlp_layer_types[layer_idx] == "sparse":
            self.mlp = Zonos3MiniSparseMoeBlock(config)
        else:
            self.mlp = Zonos3MiniMLP(config)
        self.depth_router = Zonos3MiniDepthRouter(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        sources: torch.Tensor,
        mlp_query: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        cache: Cache | None = None,
        position_ids: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        hidden_states (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            The token mixer's input, routed over depth from `sources`.
        sources (`torch.Tensor` of shape `(num_sources, batch_size, sequence_length, hidden_size)`):
            The embeddings and the outputs of every earlier sublayer.
        mlp_query (`torch.Tensor` of shape `(hidden_size,)`):
            The depth-routing query of the feed-forward's input.

        Returns the outputs of the token mixer and of the feed-forward.
        """
        hidden_states = self.input_layernorm(hidden_states)
        if self.layer_type == "linear_attention":
            mixer_output = self.linear_attn(hidden_states, cache=cache, attention_mask=attention_mask, **kwargs)
        else:
            mixer_output, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cache=cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        mlp_input = self.depth_router(torch.cat([sources, mixer_output[None]]), mlp_query)
        mlp_output = self.mlp(self.post_attention_layernorm(mlp_input))
        return mixer_output, mlp_output


class Zonos3MiniEmbeddings(CsmBackboneModelEmbeddings):
    """Sum of the 10 audio columns' and the text column's embeddings; the speaker and condition rows are replaced."""

    def __init__(self, config: Zonos3MiniConfig):
        super().__init__(config)
        self.num_codebooks = config.num_codebooks
        self.embed_audio_tokens = nn.Embedding(config.num_codebooks * config.audio_vocab_size, config.hidden_size)
        self.audio_tokens_offsets = nn.Buffer(
            torch.arange(config.num_codebooks) * config.audio_vocab_size, persistent=False
        )
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_conditions = nn.Embedding(
            sum(count + 1 for count in config.condition_bucket_counts), config.hidden_size
        )
        self.speaker_proj = nn.Linear(config.speaker_embedding_size, config.hidden_size)

    def forward(
        self,
        input_ids: torch.LongTensor,
        speaker_embeddings: torch.FloatTensor | None = None,
        speaker_token_mask: torch.BoolTensor | None = None,
        condition_ids: torch.LongTensor | None = None,
    ) -> torch.Tensor:
        audio_ids = input_ids[..., : self.num_codebooks] + self.audio_tokens_offsets
        inputs_embeds = self.embed_audio_tokens(audio_ids).sum(-2) + self.embed_tokens(input_ids[..., -1])
        if speaker_embeddings is not None and speaker_token_mask is not None:
            speaker_embeds = self.speaker_proj(speaker_embeddings.to(self.speaker_proj.weight.dtype))
            inputs_embeds = inputs_embeds.masked_scatter(
                speaker_token_mask[..., None].bool(), speaker_embeds.to(inputs_embeds.dtype)
            )
        if condition_ids is not None:
            condition_embeds = self.embed_conditions(condition_ids.clamp(min=0))
            inputs_embeds = torch.where((condition_ids >= 0)[..., None], condition_embeds, inputs_embeds)
        return inputs_embeds


@auto_docstring
class Zonos3MiniPreTrainedModel(PreTrainedModel):
    config: Zonos3MiniConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Zonos3MiniDecoderLayer", "Zonos3MiniSpeakerEncoderMECTBlock"]
    _keep_in_fp32_modules_strict = ["speaker_encoder", "balancing_biases"]
    _keys_to_ignore_on_load_missing = [r"num_batches_tracked"]
    _skip_keys_device_placement = ["cache"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _is_stateful = True
    _can_compile_fullgraph = True
    _can_record_outputs = {
        "hidden_states": OutputRecorder(Zonos3MiniDecoderLayer, index=1),
        "attentions": Zonos3MiniAttention,
        "router_logits": OutputRecorder(Zonos3MiniTopKRouter, index=0),
    }

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        std = getattr(self.config, "initializer_range", 0.02)
        if isinstance(module, Zonos3MiniGatedDeltaNet):
            init.ones_(module.dt_bias)
            # Lower bound kept away from 0 so log(A) never becomes -inf
            init.copy_(
                module.A_log,
                torch.empty(module.num_v_heads, device=module.A_log.device).uniform_(0.01, 16).log_(),
            )
        elif isinstance(module, Zonos3MiniAttention):
            init.ones_(module.temperature)
        elif isinstance(module, Zonos3MiniTopKRouter):
            init.normal_(module.weight, mean=0.0, std=std)
            init.zeros_(module.balancing_biases)
        elif isinstance(module, Zonos3MiniExperts):
            init.normal_(module.gate_up_proj, mean=0.0, std=std)
            init.normal_(module.down_proj, mean=0.0, std=std)
        elif isinstance(module, Zonos3MiniEmbeddings):
            init.copy_(
                module.audio_tokens_offsets,
                torch.arange(self.config.num_codebooks) * self.config.audio_vocab_size,
            )
        elif isinstance(module, Zonos3MiniModel):
            init.zeros_(module.depth_queries)
        elif isinstance(module, Zonos3MiniSpeakerEncoderModel):
            init.zeros_(module.stage_weights)


@auto_docstring(
    custom_intro="""
    The ZONOS3-Mini speaker encoder (MECT-B2 backbone with attentive statistics pooling): Kaldi filterbank features of
    a voice clip to a 192-dimensional speaker embedding.
    """
)
class Zonos3MiniSpeakerEncoderModel(Zonos3MiniPreTrainedModel):
    config: Zonos3MiniSpeakerEncoderConfig
    main_input_name = "input_features"
    input_modalities = ("audio",)
    _can_record_outputs = {}

    def __init__(self, config: Zonos3MiniSpeakerEncoderConfig):
        super().__init__(config)
        self.stem_conv = nn.Conv2d(1, config.in_channels, 3, padding=1)
        self.stem_norm = nn.BatchNorm2d(config.in_channels, eps=config.batch_norm_eps)
        channels, freq_bins, stages = config.in_channels, config.num_mel_bins, []
        for stage_specs in config.stage_block_specs:
            blocks = []
            for num_res_blocks, out_channels, freq_stride, time_stride, expansion in stage_specs:
                blocks.append(
                    Zonos3MiniSpeakerEncoderMECTBlock(
                        config,
                        num_res_blocks,
                        channels,
                        out_channels,
                        freq_bins,
                        (freq_stride, time_stride),
                        expansion,
                    )
                )
                channels, freq_bins = out_channels, freq_bins // freq_stride
            stages.append(nn.ModuleList(blocks))
        self.stages = nn.ModuleList(stages)
        flat_size = config.in_channels * config.num_mel_bins
        self.stage_weights = nn.Parameter(torch.zeros(1, len(stages), flat_size, 1))
        self.pooling = Zonos3MiniSpeakerEncoderStatisticsPooling(config)
        self.norm = nn.BatchNorm1d(2 * flat_size, eps=config.batch_norm_eps)
        self.proj = nn.Linear(2 * flat_size, config.embedding_size, bias=False)
        self.post_init()

    def _embed(self, features: torch.Tensor) -> torch.Tensor:
        """`features` `[1, num_mel_bins, num_frames]` -> `[1, embedding_size]`."""
        hidden_states = self.stem_norm(self.stem_conv(features[:, None]))
        stage_outputs = []
        for stage in self.stages:
            for block in stage:
                hidden_states = block(hidden_states)
            stage_outputs.append(
                hidden_states.permute(0, 2, 1, 3).reshape(hidden_states.shape[0], -1, hidden_states.shape[-1])
            )
        stacked = torch.stack(stage_outputs, dim=1)
        pooled = self.pooling((F.softmax(self.stage_weights, dim=1) * stacked).sum(dim=1))
        return self.proj(self.norm(pooled))

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_features: torch.FloatTensor,
        attention_mask: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Zonos3MiniSpeakerEncoderOutput:
        r"""
        input_features (`torch.FloatTensor` of shape `(batch_size, num_frames, num_mel_bins)`):
            Mean-normalized Kaldi filterbank features of the voice clips (see [`Zonos3MiniFeatureExtractor`]).
        attention_mask (`torch.LongTensor` of shape `(batch_size, num_frames)`, *optional*):
            Mask of the valid frames of `input_features`.
        """
        # The backbone has no padding mask and its time-strided blocks repeat frames back to the input length, so a
        # padded batch would not be exact: each clip runs on its own valid frames.
        clips = (
            torch.split(input_features[attention_mask.bool()], attention_mask.sum(-1).tolist())
            if attention_mask is not None
            else input_features.unbind(0)
        )
        embeddings = [self._embed(clip.transpose(0, 1)[None].to(self.dtype)) for clip in clips]
        return Zonos3MiniSpeakerEncoderOutput(speaker_embeddings=torch.cat(embeddings).float())


@auto_docstring
class Zonos3MiniModel(Zonos3MiniPreTrainedModel):
    def __init__(self, config: Zonos3MiniConfig):
        super().__init__(config)
        self.embeddings = Zonos3MiniEmbeddings(config)
        self.embed_norm = Zonos3MiniUnweightedRMSNorm(config.rms_norm_eps)
        self.layers = nn.ModuleList(
            [Zonos3MiniDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Zonos3MiniRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Zonos3MiniRotaryEmbedding(config=config)
        self.depth_router = Zonos3MiniDepthRouter(config)
        self.depth_queries = nn.Parameter(torch.zeros(2 * config.num_hidden_layers + 1, config.hidden_size))
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        # the row embedder: it maps rows of audio and text ids to embeddings
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        cache: Cache | None = None,
        use_cache: bool | None = None,
        speaker_embeddings: torch.FloatTensor | None = None,
        speaker_token_mask: torch.BoolTensor | None = None,
        condition_ids: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeModelOutputWithPast:
        r"""
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length, num_codebooks + 1)`):
            The prompt rows: the audio columns (delayed codec codes) and the text column (see [`Zonos3MiniProcessor`]).
        speaker_embeddings (`torch.FloatTensor` of shape `(batch_size, speaker_embedding_size)`, *optional*):
            The speaker embeddings projected into the rows flagged by `speaker_token_mask`.
        speaker_token_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length)`, *optional*):
            The speaker rows, one per item that has a speaker embedding.
        condition_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            The condition bucket of each condition row (offset into the shared condition table), `-1` elsewhere.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids, speaker_embeddings, speaker_token_mask, condition_ids)

        if use_cache and cache is None:
            cache = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = cache.get_seq_length() if cache is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache": cache,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "linear_attention": create_recurrent_attention_mask(**mask_kwargs),
            }

        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
        sources = self.embed_norm(inputs_embeds)[None]
        for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            hidden_states = self.depth_router(sources, self.depth_queries[2 * layer_idx])
            mixer_output, mlp_output = decoder_layer(
                hidden_states,
                sources,
                self.depth_queries[2 * layer_idx + 1],
                attention_mask=causal_mask_mapping[self.config.layer_types[layer_idx]],
                position_embeddings=position_embeddings,
                cache=cache,
                position_ids=position_ids,
                **kwargs,
            )
            sources = torch.cat([sources, mixer_output[None], mlp_output[None]])

        hidden_states = self.norm(self.depth_router(sources, self.depth_queries[-1]))
        return MoeModelOutputWithPast(last_hidden_state=hidden_states, cache=cache)


@auto_docstring(
    custom_intro="""
    ZONOS3-Mini: a hybrid gated DeltaNet / attention MoE language model over rows of 10 delayed ZAC-1 codebooks and a
    text column, with a speaker encoder for voice cloning from a speaker embedding.
    """
)
class Zonos3MiniForConditionalGeneration(Zonos3MiniPreTrainedModel, Zonos3MiniGenerationMixin):
    output_modalities = ("audio",)

    def __init__(self, config: Zonos3MiniConfig):
        super().__init__(config)
        self.model = Zonos3MiniModel(config)
        self.speaker_encoder = Zonos3MiniSpeakerEncoderModel(config.speaker_encoder_config)
        self.lm_head = nn.Linear(config.hidden_size, config.num_codebooks * config.audio_vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        cache: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        speaker_embeddings: torch.FloatTensor | None = None,
        speaker_input_features: torch.FloatTensor | None = None,
        speaker_input_features_mask: torch.LongTensor | None = None,
        speaker_token_mask: torch.BoolTensor | None = None,
        condition_ids: torch.LongTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Zonos3MiniOutputWithPast:
        r"""
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length, num_codebooks + 1)`):
            The prompt rows: the audio columns (delayed codec codes) and the text column (see [`Zonos3MiniProcessor`]).
        speaker_embeddings (`torch.FloatTensor` of shape `(batch_size, speaker_embedding_size)`, *optional*):
            Precomputed speaker embeddings for the rows flagged by `speaker_token_mask`.
        speaker_input_features (`torch.FloatTensor` of shape `(batch_size, num_frames, num_mel_bins)`, *optional*):
            Filterbank features of voice clips, embedded by the speaker encoder when `speaker_embeddings` is not given.
        speaker_input_features_mask (`torch.LongTensor` of shape `(batch_size, num_frames)`, *optional*):
            Mask of the valid frames of `speaker_input_features`.
        speaker_token_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length)`, *optional*):
            The speaker rows, one per item that has a speaker embedding.
        condition_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            The condition bucket of each condition row (offset into the shared condition table), `-1` elsewhere.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length, num_codebooks)`, *optional*):
            Audio codes to predict at each row (the next row's audio columns are predicted from each row); `-100`
            entries are ignored.

        Example:

        ```python
        >>> from transformers import AutoModelForTextToWaveform, AutoProcessor

        >>> model_id = "Zyphra/ZONOS3-Mini-transformers"
        >>> processor = AutoProcessor.from_pretrained(model_id)
        >>> model = AutoModelForTextToWaveform.from_pretrained(model_id)

        >>> conversation = [{"role": "user", "content": "Hello from ZONOS3-Mini."}]
        >>> inputs = processor.apply_chat_template(conversation, tokenize=True, return_dict=True)
        >>> outputs = model(**inputs)
        >>> logits = outputs.logits  # (batch_size, sequence_length, num_codebooks, codebook_size + 2)
        ```"""
        if speaker_embeddings is None and speaker_input_features is not None:
            speaker_embeddings = self.speaker_encoder(
                speaker_input_features, attention_mask=speaker_input_features_mask
            ).speaker_embeddings

        outputs: MoeModelOutputWithPast = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache=cache,
            use_cache=use_cache,
            speaker_embeddings=speaker_embeddings,
            speaker_token_mask=speaker_token_mask,
            condition_ids=condition_ids,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        logits = logits.view(*logits.shape[:2], self.config.num_codebooks, self.config.audio_vocab_size)
        softcap = self.config.final_logit_softcapping
        logits = softcap * torch.tanh(logits / softcap)

        loss = None
        if labels is not None:
            # row `t` predicts the audio columns of row `t + 1`
            loss = self.loss_function(
                logits=logits[:, :-1].contiguous(),
                labels=None,
                vocab_size=self.config.audio_vocab_size,
                shift_labels=labels[:, 1:].contiguous(),
                **kwargs,
            )

        return Zonos3MiniOutputWithPast(
            loss=loss,
            logits=logits,
            cache=outputs.cache,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )


@requires(backends=("torch", "torchaudio"))
class Zonos3MiniFeatureExtractor(Speech2TextFeatureExtractor):
    r"""
    Constructs a ZONOS3-Mini speaker feature extractor: the voice clip (mixed to mono) is resampled to 16 kHz, cut to
    `max_seconds` and turned into Kaldi filterbank features (Hamming window, no dither, no energy floor) with the
    per-utterance mean removed, the input of [`Zonos3MiniSpeakerEncoderModel`].

    Args:
        feature_size (`int`, *optional*, defaults to 80):
            The feature dimension of the extracted features.
        sampling_rate (`int`, *optional*, defaults to 44100):
            The sampling rate of the input audio.
        num_mel_bins (`int`, *optional*, defaults to 80):
            Number of mel filterbank bins.
        padding_value (`float`, *optional*, defaults to 0.0):
            The value used to pad the features.
        speaker_sampling_rate (`int`, *optional*, defaults to 16000):
            The sampling rate the features are computed at.
        max_seconds (`float`, *optional*, defaults to 60.0):
            Longer clips are cut to this duration.
        waveform_scale (`float`, *optional*, defaults to 32768.0):
            Scale applied to the waveform before the filterbank (Kaldi expects 16-bit integer amplitudes).
        normalize_means (`bool`, *optional*, defaults to `True`):
            Whether to remove the per-utterance mean of the features.
        normalize_vars (`bool`, *optional*, defaults to `False`):
            Whether to normalize the per-utterance variance of the features.
    """

    def __init__(
        self,
        feature_size=80,
        sampling_rate=44100,
        num_mel_bins=80,
        padding_value=0.0,
        speaker_sampling_rate=16000,
        max_seconds=60.0,
        waveform_scale=32768.0,
        normalize_means=True,
        normalize_vars=False,
        **kwargs,
    ):
        SequenceFeatureExtractor.__init__(
            self, feature_size=feature_size, sampling_rate=sampling_rate, padding_value=padding_value, **kwargs
        )
        self.num_mel_bins = num_mel_bins
        self.normalize_means = normalize_means
        self.normalize_vars = normalize_vars
        self.return_attention_mask = True
        self.speaker_sampling_rate = speaker_sampling_rate
        self.max_seconds = max_seconds
        self.waveform_scale = waveform_scale
        self._resampler = None

    def _extract_fbank_features(self, waveform: "torch.Tensor") -> "torch.Tensor":
        """Kaldi filterbank of a 16 kHz float32 waveform `[num_samples]` -> `[num_frames, num_mel_bins]`."""
        if waveform.shape[-1] < 400:
            raise ValueError(
                f"The voice clip is too short: {waveform.shape[-1]} samples at {self.speaker_sampling_rate} Hz "
                "(at least one 25 ms frame)."
            )
        return torchaudio.compliance.kaldi.fbank(
            waveform[None] * self.waveform_scale,
            num_mel_bins=self.num_mel_bins,
            sample_frequency=self.speaker_sampling_rate,
            window_type="hamming",
            dither=0.0,
            energy_floor=0.0,
        )

    def __call__(
        self,
        audio: AudioInput,
        sampling_rate: int | None = None,
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ) -> BatchFeature:
        """
        Args:
            audio (`np.ndarray`, `torch.Tensor`, `list[np.ndarray]`, `list[torch.Tensor]`):
                One voice clip of shape `(num_samples,)` or `(channels, num_samples)`, or a list of clips.
            sampling_rate (`int`, *optional*):
                The sampling rate of `audio`; it must be `self.sampling_rate`.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                The type of tensors to return (`"pt"` or `"np"`).
        """
        if sampling_rate is not None and sampling_rate != self.sampling_rate:
            raise ValueError(
                f"{self.__class__.__name__} expects audio at {self.sampling_rate} Hz, got {sampling_rate} Hz."
            )
        if self._resampler is None:
            # the float64-built kernel of `torchaudio.transforms.Resample`, which the original's resampler reproduces
            self._resampler = torchaudio.transforms.Resample(self.sampling_rate, self.speaker_sampling_rate)
        max_samples = int(self.max_seconds * self.speaker_sampling_rate)
        features = []
        for clip in make_list_of_audio(audio):
            clip = torch.as_tensor(np.asarray(clip, dtype=np.float32))
            clip = clip.mean(0) if clip.ndim == 2 else clip
            waveform = self._resampler(clip)[:max_samples]
            fbank = self._extract_fbank_features(waveform)
            if self.normalize_means:
                fbank = fbank - fbank.mean(0, keepdim=True)
            if self.normalize_vars:
                fbank = fbank / fbank.std(0, keepdim=True)
            features.append(fbank)
        input_features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True, padding_value=self.padding_value)
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [torch.ones(len(fbank), dtype=torch.long) for fbank in features], batch_first=True
        )
        return BatchFeature(
            {"input_features": input_features, "attention_mask": attention_mask}, tensor_type=return_tensors
        )

    def to_dict(self):
        output = super().to_dict()
        output.pop("_resampler", None)
        return output


__all__ = [
    "Zonos3MiniConfig",
    "Zonos3MiniFeatureExtractor",
    "Zonos3MiniSpeakerEncoderConfig",
    "Zonos3MiniPreTrainedModel",
    "Zonos3MiniSpeakerEncoderModel",
    "Zonos3MiniModel",
    "Zonos3MiniForConditionalGeneration",
]
