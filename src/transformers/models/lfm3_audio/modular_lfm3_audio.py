# Copyright 2026 The HuggingFace Team. All rights reserved.
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
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from huggingface_hub.dataclasses import strict
from torch import nn

from ... import initialization as init
from ...cache_utils import Cache, DynamicCache
from ...configuration_utils import PreTrainedConfig, SubConfigSpec
from ...masking_utils import (
    create_bidirectional_mask,
    create_causal_mask,
    create_recurrent_attention_mask,
    create_sliding_window_causal_mask,
)
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPast, CausalLMOutputWithPast, MaskedLMOutput
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, slice_logits_to_keep
from ...processing_utils import Unpack
from ...utils import (
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
    torch_compilable_check,
)
from ...utils.deprecation import deprecate_kwarg
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..auto import AutoConfig, AutoModel
from ..higgs_audio_v2.modeling_higgs_audio_v2 import HiggsAudioV2Embeddings
from ..internvl.modeling_internvl import InternVLMultiModalProjector
from ..lfm3.configuration_lfm3 import Lfm3Config
from ..lfm3.modeling_lfm3 import Lfm3Attention, Lfm3DecoderLayer, Lfm3Model, Lfm3PreTrainedModel
from ..liquid_codec.modeling_liquid_codec import LiquidCodecRotaryEmbedding
from ..llama.modeling_llama import eager_attention_forward
from ..nemotron_asr_streaming.modeling_nemotron_asr_streaming import (
    NemotronAsrStreamingEncoderCausalConv2D,
    NemotronAsrStreamingEncoderCausalConv2dCacheLayer,
    NemotronAsrStreamingEncoderCausalConvPaddingCache,
    NemotronAsrStreamingEncoderModelOutput,
    NemotronAsrStreamingEncoderSubsamplingConv2D,
    NemotronAsrStreamingEncoderSubsamplingLayer,
    _mask_subsampled_frames,
)
from ..olmo.modeling_olmo import rotate_half
from ..olmo2.modeling_olmo2 import Olmo2RMSNorm
from .generation_lfm3_audio import Lfm3AudioGenerationMixin


@auto_docstring(checkpoint="LiquidAI/LFM3-Audio-2B")
@strict
class Lfm3AudioAudioConfig(Lfm3Config):
    r"""
    conv_bias (`bool`, *optional*, defaults to `False`):
        Whether to use bias in the conv layers.
    conv_L_cache (`int`, *optional*, defaults to 3):
        L_cache dim in the conv layers.
    use_sandwich_norm (`bool`, *optional*, defaults to `False`):
        Whether to apply an RMSNorm to the operator and feed-forward outputs before they enter the residual stream.
    use_attention_gate (`bool`, *optional*, defaults to `False`):
        Whether the attention layers scale each head's output by a sigmoid gate computed from the layer input.
    num_mel_bins (`int`, *optional*, defaults to 128):
        The number of mel bins of the input spectrogram.
    subsampling_factor (`int`, *optional*, defaults to 8):
        The number of spectrogram frames per encoder token, a power of 2: one strided convolution per factor of 2.
    subsampling_conv_channels (`int`, *optional*, defaults to 256):
        The number of channels of the subsampling convolutions.
    subsampling_conv_kernel_size (`int`, *optional*, defaults to 3):
        The kernel size of the subsampling convolutions.
    subsampling_conv_stride (`int`, *optional*, defaults to 2):
        The stride of the subsampling convolutions.
    """

    model_type = "lfm3_audio_audio"
    base_config_key = "audio_config"

    vocab_size = AttributeError()
    pad_token_id = AttributeError()
    bos_token_id = AttributeError()
    eos_token_id = AttributeError()
    tie_word_embeddings = AttributeError()
    base_model_tp_plan = AttributeError()

    hidden_size: int = 768
    intermediate_size: int = 1792
    num_hidden_layers: int = 17
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    head_dim: int = 96
    sliding_window: int | None = 32
    conv_L_cache: int = 3
    use_sandwich_norm: bool = False
    use_attention_gate: bool = False
    num_mel_bins: int = 128
    subsampling_factor: int = 8
    subsampling_conv_channels: int = 256
    subsampling_conv_kernel_size: int = 3
    subsampling_conv_stride: int = 2

    def __post_init__(self, **kwargs):
        if self.rope_parameters is None:
            self.rope_parameters = {"sliding_attention": {"rope_type": "default", "rope_theta": 1_000_000.0}}

        if self.layer_types is None:
            self.layer_types = ["conv", "conv", "sliding_attention"] * (self.num_hidden_layers // 3) + [
                "sliding_attention"
            ] * (self.num_hidden_layers % 3)

        super().__post_init__(**kwargs)

    @property
    def subsampling_out_hidden_size(self) -> int:
        # the mel axis is padded by `(kernel_size - 1) // 2` on both sides
        padding = 2 * ((self.subsampling_conv_kernel_size - 1) // 2)
        num_bins = self.num_mel_bins
        for _ in range(int(math.log2(self.subsampling_factor))):
            num_bins = (num_bins + padding - self.subsampling_conv_kernel_size) // self.subsampling_conv_stride + 1

        return self.subsampling_conv_channels * num_bins

    def validate_architecture(self):
        if self.subsampling_factor < 2 or self.subsampling_factor & (self.subsampling_factor - 1):
            raise ValueError(f"`subsampling_factor` must be a power of 2, got {self.subsampling_factor}")


@auto_docstring(checkpoint="LiquidAI/LFM3-Audio-2B")
@strict
class Lfm3AudioDepthDecoderConfig(Lfm3Config):
    r"""
    conv_bias (`bool`, *optional*, defaults to `False`):
        Whether to use bias in the conv layers.
    conv_L_cache (`int`, *optional*, defaults to 7):
        L_cache dim in the conv layers.
    use_sandwich_norm (`bool`, *optional*, defaults to `False`):
        Whether to apply an RMSNorm to the operator and feed-forward outputs before they enter the residual stream.
    use_attention_gate (`bool`, *optional*, defaults to `True`):
        Whether the attention layers scale each head's output by a sigmoid gate computed from the layer input.
    num_codebooks (`int`, *optional*, defaults to 12):
        The number of codebooks of a speech frame, one input slot each.
    codebook_size (`int`, *optional*, defaults to 2017):
        The size of each codebook. The embedding tables of all codebooks are stored as one
        `(num_codebooks, codebook_size, hidden_size)` weight.
    backbone_hidden_size (`int`, *optional*, defaults to 2048):
        The hidden size of the language model, whose hidden state the decoder projects into its input slots.
    """

    model_type = "lfm3_audio_depth_decoder"
    base_config_key = "depth_decoder_config"

    vocab_size = AttributeError()
    pad_token_id = AttributeError()
    bos_token_id = AttributeError()
    eos_token_id = AttributeError()
    base_model_tp_plan = AttributeError()

    hidden_size: int = 1024
    intermediate_size: int = 2816
    num_hidden_layers: int = 3
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 32
    max_position_embeddings: int = 12
    sliding_window: int | None = None
    use_sandwich_norm: bool = False
    use_cache: bool = False
    num_codebooks: int = 12
    codebook_size: int = 2017
    backbone_hidden_size: int = 2048

    def __post_init__(self, **kwargs):
        if self.rope_parameters is None:
            self.rope_parameters = {
                "full_attention": {"rope_type": "default", "rope_theta": 1_000_000.0, "partial_rotary_factor": 1.0}
            }

        if self.layer_types is None:
            self.layer_types = ["full_attention"] * self.num_hidden_layers

        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="LiquidAI/LFM3-Audio-2B")
@strict
class Lfm3AudioConfig(PreTrainedConfig):
    r"""
    audio_config (`Lfm3AudioAudioConfig` or `dict`, *optional*):
        Speech encoder config. `None`: no speech encoder nor projector, no speech input.
    depth_decoder_config (`Lfm3AudioDepthDecoderConfig` or `dict`, *optional*):
        Depth decoder config (bidirectional LFM3 model over the codebooks of a speech frame). `None`: no depth
        decoder nor speech frame embeddings, no speech frames in or out.
    num_codebooks (`int`, *optional*, defaults to 12):
        The number of codebooks of a speech frame.
    codebook_size (`int`, *optional*, defaults to 2017):
        The size of each codebook, the codec's codes plus the end-of-audio code.
    audio_eos_code (`int`, *optional*, defaults to 2016):
        The end-of-audio code. A frame holding it in every codebook ends a speech segment.
    audio_out_token_id (`int`, *optional*, defaults to 125032):
        The id of the placeholder token standing for one output speech frame.
    audio_start_token_id (`int`, *optional*, defaults to 125012):
        The id of `<|audio_start|>`, which switches generation to speech until an end-of-audio frame.
    text_end_token_id (`int`, *optional*, defaults to 125014):
        The id of `<|text_end|>`, which ends the text of an interleaved answer.
    mixed_start_token_id (`int`, *optional*, defaults to 125015):
        The id of `<|mixed_start|>`, which switches generation to interleaved text and speech.
    mixed_end_token_id (`int`, *optional*, defaults to 125016):
        The id of `<|mixed_end|>`, which opens the closing text of a live transcription after its last audio.
    interleaved_n_text (`int`, *optional*, defaults to 6):
        The number of text tokens of a block of an interleaved answer.
    interleaved_n_audio (`int`, *optional*, defaults to 12):
        Speech frames per block of an interleaved answer; also encoder tokens per block of a live transcription.
    maskgit_schedule (`list[int]`, *optional*):
        Codebooks revealed per depth decoder pass, summing to `num_codebooks`. Defaults to three equal passes.
    projector_hidden_act (`str`, *optional*, defaults to `"gelu"`):
        The activation of the projector from the speech encoder to the language model.
    text_loss_weight (`float`, *optional*, defaults to 1.0):
        The weight of the text loss in the training loss.
    audio_loss_weight (`float`, *optional*, defaults to 1.0):
        The weight of the speech loss in the training loss.
    """

    model_type = "lfm3_audio"
    sub_configs_defaults = {
        "text_config": SubConfigSpec(config_class=AutoConfig, model_type="lfm3"),
        "audio_config": SubConfigSpec(config_class=Lfm3AudioAudioConfig, optional=True),
        "depth_decoder_config": SubConfigSpec(config_class=Lfm3AudioDepthDecoderConfig, optional=True),
    }

    text_config: dict | PreTrainedConfig | None = None
    audio_config: dict | PreTrainedConfig | None = field(default_factory=dict)
    depth_decoder_config: dict | PreTrainedConfig | None = field(default_factory=dict)
    num_codebooks: int = 12
    codebook_size: int = 2017
    audio_eos_code: int = 2016
    audio_token_id: int = 125_031
    audio_out_token_id: int = 125_032
    audio_start_token_id: int = 125_012
    text_end_token_id: int = 125_014
    mixed_start_token_id: int = 125_015
    mixed_end_token_id: int = 125_016
    interleaved_n_text: int = 6
    interleaved_n_audio: int = 12
    maskgit_schedule: list[int] | None = None
    tie_word_embeddings: bool = True
    projector_hidden_act: str = "gelu"
    text_loss_weight: float = 1.0
    audio_loss_weight: float = 1.0
    initializer_range: float = 0.02

    def __post_init__(self, **kwargs):
        if self.maskgit_schedule is None:
            third = self.num_codebooks // 3
            self.maskgit_schedule = [third, third, self.num_codebooks - 2 * third]

        super().__post_init__(**kwargs)

    def validate_architecture(self):
        if sum(self.maskgit_schedule) != self.num_codebooks:
            raise ValueError(
                f"`maskgit_schedule` must reveal all {self.num_codebooks} codebooks, got {self.maskgit_schedule}."
            )

        depth = self.depth_decoder_config
        if depth is not None and (depth.num_codebooks, depth.codebook_size, depth.backbone_hidden_size) != (
            self.num_codebooks,
            self.codebook_size,
            self.text_config.hidden_size,
        ):
            raise ValueError(
                "`depth_decoder_config` must match `num_codebooks`, `codebook_size` and the language model's "
                f"`hidden_size`, got {depth.num_codebooks}, {depth.codebook_size} and {depth.backbone_hidden_size}."
            )


class Lfm3AudioRotaryEmbedding(LiquidCodecRotaryEmbedding):
    pass


class Lfm3AudioAttention(Lfm3Attention):
    pass


class Lfm3AudioDecoderLayer(Lfm3DecoderLayer):
    pass


class Lfm3AudioEncoderAttention(Lfm3Attention):
    """RoPE in the model dtype: the speech encoder is an LFM2 model in the original."""

    def rotate(self, query_states, key_states, cos, sin):
        cos, sin = cos.unsqueeze(1).to(query_states.dtype), sin.unsqueeze(1).to(query_states.dtype)
        query_states = (query_states * cos) + (rotate_half(query_states) * sin)
        key_states = (key_states * cos) + (rotate_half(key_states) * sin)
        return query_states, key_states

    @deprecate_kwarg("past_key_values", "5.40.0", new_name="cache")
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
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        query_states, key_states = self.rotate(query_states, key_states, *position_embeddings)

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
            sliding_window=self.sliding_window,
            **kwargs,
        )
        if self.g_proj is not None:
            attn_output = attn_output * torch.sigmoid(self.g_proj(hidden_states)).unsqueeze(-1)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Lfm3AudioEncoderLayer(Lfm3DecoderLayer):
    def __init__(self, config: Lfm3AudioAudioConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        if self.is_attention_layer:
            self.self_attn = Lfm3AudioEncoderAttention(config, layer_idx)


@auto_docstring
class Lfm3AudioPreTrainedModel(Lfm3PreTrainedModel):
    config: Lfm3AudioConfig
    base_model_prefix = "model"
    input_modalities = ("audio", "text")
    _no_split_modules = [
        "Lfm3AudioDecoderLayer",
        "Lfm3AudioEncoderLayer",
        "Lfm3AudioEncoderSubsamplingConv2D",
        "Lfm3AudioDepthDecoderModel",
    ]

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, Lfm3AudioEmbeddings):
            init.copy_(
                module.audio_tokens_offsets, torch.arange(self.config.num_codebooks) * self.config.codebook_size
            )

        if isinstance(module, Lfm3AudioDepthDecoderEmbeddings):
            init.normal_(module.weight, std=self.config.initializer_range)
            init.normal_(module.mask_embedding, std=self.config.initializer_range)
            init.copy_(module.codebook_offsets, torch.arange(module.num_codebooks) * module.codebook_size)

        if isinstance(module, Lfm3AudioDepthDecoderHead):
            init.normal_(module.weight, std=self.config.initializer_range)


class Lfm3AudioEncoderCausalConv2dCacheLayer(NemotronAsrStreamingEncoderCausalConv2dCacheLayer): ...


class Lfm3AudioEncoderCausalConvPaddingCache(NemotronAsrStreamingEncoderCausalConvPaddingCache):
    def __init__(self):
        self.layers: dict[str, Lfm3AudioEncoderCausalConv2dCacheLayer] = {}

    def update(self, hidden_states, cache_key, conv_module):
        if cache_key not in self.layers:
            self.layers[cache_key] = Lfm3AudioEncoderCausalConv2dCacheLayer()

        padding_states = self.layers[cache_key].update(hidden_states, conv_module)
        return torch.cat([padding_states, hidden_states], dim=2)


class Lfm3AudioEncoderCausalConv2D(NemotronAsrStreamingEncoderCausalConv2D):
    # centered padding, unlike Nemotron
    @property
    def left_pad_init(self):
        return self.left_pad

    @property
    def time_pad(self):
        return (self.left_pad, 0)

    @property
    def freq_pad(self):
        return ((self.kernel_size[1] - 1) // 2, (self.kernel_size[1] - 1) // 2)

    def output_length(self, input_lengths: torch.Tensor | None, streaming: bool = False) -> torch.Tensor | None:
        if input_lengths is None:
            return None

        return -(input_lengths // -self.stride[0])

    def forward(
        self, x: torch.Tensor, padding_cache: Lfm3AudioEncoderCausalConvPaddingCache | None = None
    ) -> torch.Tensor:
        # last chunk of a stream: right-pad to a multiple of the stride, as offline
        x = nn.functional.pad(x, (0, 0, 0, -x.shape[2] % self.stride[0]))
        return super().forward(x, padding_cache)


class Lfm3AudioEncoderSubsamplingLayer(NemotronAsrStreamingEncoderSubsamplingLayer):
    def __init__(self, config: Lfm3AudioAudioConfig, layer_idx: int):
        super().__init__(config, layer_idx)


class Lfm3AudioEncoderSubsamplingConv2D(NemotronAsrStreamingEncoderSubsamplingConv2D):
    def __init__(self, config: Lfm3AudioAudioConfig):
        super().__init__(config)
        self.act_fn = nn.SiLU()
        # a pointwise convolution in place of Nemotron's linear
        self.conv_out = nn.Conv1d(config.subsampling_out_hidden_size, config.hidden_size, kernel_size=1)

        del self.linear

    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor = None,
        padding_cache: Lfm3AudioEncoderCausalConvPaddingCache | None = None,
    ):
        hidden_states = input_features.unsqueeze(1)
        lengths = attention_mask.sum(-1) if attention_mask is not None else None

        # stem stage
        hidden_states = self.conv_in(hidden_states, padding_cache=padding_cache)
        lengths = self.conv_in.output_length(lengths, streaming=padding_cache is not None)
        hidden_states = self.act_fn(_mask_subsampled_frames(hidden_states, lengths))

        # depthwise-separable stages
        for layer in self.layers:
            hidden_states, lengths = layer(hidden_states, lengths, padding_cache=padding_cache)
            hidden_states = self.act_fn(hidden_states)

        hidden_states = hidden_states.transpose(1, 2).flatten(2)
        hidden_states = self.conv_out(hidden_states.transpose(1, 2)).transpose(1, 2)

        return hidden_states


@auto_docstring(
    custom_intro="""
    Base class for the LFM3-Audio speech encoder outputs, with the streaming states.
    """
)
@dataclass
class Lfm3AudioAudioModelOutput(NemotronAsrStreamingEncoderModelOutput):
    r"""
    pooler_output (`torch.FloatTensor`, *optional*):
        Set by [`Lfm3AudioModel.get_audio_features`]: the projected valid tokens of all clips, flattened.
    attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
        Mask of the valid tokens, after subsampling.
    cache (`Cache`, *optional*):
        Encoder cache, for the next chunk.
    padding_cache (`Lfm3AudioEncoderCausalConvPaddingCache`, *optional*):
        Subsampling state, for the next chunk.
    """


@auto_docstring(
    custom_intro="""
    The LFM3-Audio speech encoder.
    """
)
class Lfm3AudioAudioModel(Lfm3Model):
    config: Lfm3AudioAudioConfig
    main_input_name = "input_features"
    input_modalities = "audio"
    _can_record_outputs = {
        "hidden_states": Lfm3AudioEncoderLayer,
        "attentions": Lfm3AudioEncoderAttention,
    }

    def __init__(self, config: Lfm3AudioAudioConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [Lfm3AudioEncoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.subsampling = Lfm3AudioEncoderSubsamplingConv2D(config)

        del self.padding_idx
        del self.vocab_size
        del self.embed_tokens

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    @deprecate_kwarg("past_key_values", "5.40.0", new_name="cache")
    def forward(
        self,
        input_features: torch.FloatTensor,
        input_features_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        cache: Cache | None = None,
        padding_cache: Lfm3AudioEncoderCausalConvPaddingCache | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Lfm3AudioAudioModelOutput:
        r"""
        input_features (`torch.FloatTensor` of shape `(batch_size, num_frames, num_mel_bins)`):
            Log-mel spectrograms, right-padded. When streaming: one chunk of a multiple of
            `config.subsampling_factor` frames, except the last.
        input_features_mask (`torch.Tensor` of shape `(batch_size, num_frames)`, *optional*):
            Mask of the valid spectrogram frames.
        padding_cache (`Lfm3AudioEncoderCausalConvPaddingCache`, *optional*):
            Subsampling streaming state: an empty one with the first chunk, then the returned one.
        """
        if use_cache:
            if cache is None:
                cache = DynamicCache(config=self.config)

            if padding_cache is None:
                padding_cache = Lfm3AudioEncoderCausalConvPaddingCache()

        inputs_embeds = self.subsampling(input_features, input_features_mask, padding_cache=padding_cache)

        past_seen_tokens = cache.get_seq_length() if cache is not None else 0
        if position_ids is None:
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        output_mask = attention_mask = None
        if input_features_mask is not None:
            output_mask = input_features_mask[:, :: self.config.subsampling_factor]
            # cached tokens are all valid
            attention_mask = nn.functional.pad(output_mask, (past_seen_tokens, 0), value=1)

        mask_kwargs = {
            "config": self.config,
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "cache": cache,
            "position_ids": position_ids,
        }
        mask_creation_functions = {
            "full_attention": lambda: create_causal_mask(**mask_kwargs),
            "sliding_attention": lambda: create_sliding_window_causal_mask(**mask_kwargs),
            "conv": lambda: create_recurrent_attention_mask(**mask_kwargs),
        }
        causal_mask_mapping = {}
        for layer_type in set(self.config.layer_types):
            causal_mask_mapping[layer_type] = mask_creation_functions[layer_type]()

        hidden_states = inputs_embeds
        position_embeddings = {
            layer_type: self.rotary_emb(hidden_states, position_ids=position_ids, layer_type=layer_type)
            for layer_type in self.rotary_emb.layer_types
        }

        for i, encoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            layer_type = self.config.layer_types[i]
            hidden_states = encoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[layer_type],
                position_embeddings=position_embeddings.get(layer_type),
                position_ids=position_ids,
                cache=cache,
                **kwargs,
            )

        hidden_states = self.embedding_norm(hidden_states)

        return Lfm3AudioAudioModelOutput(
            last_hidden_state=hidden_states,
            attention_mask=output_mask,
            cache=cache,
            padding_cache=padding_cache,
        )


class Lfm3AudioMultiModalProjector(InternVLMultiModalProjector):
    def __init__(self, config: Lfm3AudioConfig):
        super().__init__(config)
        self.layer_norm = nn.LayerNorm(config.audio_config.hidden_size)
        self.linear_1 = nn.Linear(config.audio_config.hidden_size, config.text_config.hidden_size)


class Lfm3AudioCodebookRMSNorm(Olmo2RMSNorm):
    def __init__(self, num_codebooks: int, hidden_size: int, eps: float = 1e-6):
        super().__init__(hidden_size, eps)
        self.weight = nn.Parameter(torch.ones(num_codebooks, hidden_size))


class Lfm3AudioDepthDecoderEmbeddings(nn.Module):
    """One table per codebook, stored stacked; code `codebook_size` = masked."""

    def __init__(self, config: Lfm3AudioDepthDecoderConfig):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(config.num_codebooks, config.codebook_size, config.hidden_size))
        self.num_codebooks = config.num_codebooks
        self.codebook_size = config.codebook_size
        self.codebook_offsets = nn.Buffer(torch.arange(config.num_codebooks) * config.codebook_size, persistent=False)
        self.mask_embedding = nn.Parameter(torch.empty(config.hidden_size))

    def forward(self, input_ids: torch.LongTensor) -> torch.FloatTensor:
        masked = input_ids == self.codebook_size
        input_ids = input_ids.masked_fill(masked, 0) + self.codebook_offsets.to(input_ids.device)
        embeds = nn.functional.embedding(input_ids, self.weight.flatten(0, 1))
        return torch.where(masked[..., None], self.mask_embedding, embeds)


@auto_docstring(
    custom_intro="""
    The LFM3-Audio depth decoder, a bidirectional transformer over the codebooks of a speech frame, conditioned on the
    language model's hidden state preceding the frame.
    """
)
class Lfm3AudioDepthDecoderModel(Lfm3Model):
    config: Lfm3AudioDepthDecoderConfig

    def __init__(self, config: Lfm3AudioDepthDecoderConfig):
        super().__init__(config)
        self.num_codebooks = config.num_codebooks
        self.embed_tokens = Lfm3AudioDepthDecoderEmbeddings(config)
        self.backbone_projector = nn.Linear(config.backbone_hidden_size, config.num_codebooks * config.hidden_size)

        del self.padding_idx
        del self.vocab_size

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        backbone_last_hidden_state: torch.FloatTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutput:
        r"""
        input_ids (`torch.LongTensor` of shape `(num_frames, num_codebooks)`, *optional*):
            Codes of each frame; `codebook_size` marks a masked codebook.
        backbone_last_hidden_state (`torch.FloatTensor` of shape `(num_frames, backbone_hidden_size)`):
            The language model's hidden state at the position preceding each frame.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        backbone_embeds = self.backbone_projector(backbone_last_hidden_state).unflatten(-1, (self.num_codebooks, -1))
        hidden_states = inputs_embeds + backbone_embeds
        position_ids = torch.arange(self.num_codebooks, device=hidden_states.device)[None]

        attention_mask = create_bidirectional_mask(
            config=self.config, inputs_embeds=hidden_states, attention_mask=None
        )
        position_embeddings = {
            layer_type: self.rotary_emb(hidden_states, position_ids=position_ids, layer_type=layer_type)
            for layer_type in self.rotary_emb.layer_types
        }

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            layer_type = self.config.layer_types[i]
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings.get(layer_type),
                position_ids=position_ids,
                is_causal=False,
                **kwargs,
            )

        hidden_states = self.embedding_norm(hidden_states)

        return BaseModelOutput(last_hidden_state=hidden_states)


class Lfm3AudioDepthDecoderHead(nn.Module):
    """Per-codebook RMSNorm + linear head, stored as one `(num_codebooks, codebook_size, hidden_size)` weight."""

    def __init__(self, config: Lfm3AudioDepthDecoderConfig):
        super().__init__()
        self.num_codebooks = config.num_codebooks
        self.codebook_size = config.codebook_size
        self.norm = Lfm3AudioCodebookRMSNorm(config.num_codebooks, config.hidden_size, eps=config.norm_eps)
        self.weight = nn.Parameter(torch.empty(config.num_codebooks, config.codebook_size, config.hidden_size))

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        hidden_states = self.norm(hidden_states)
        return torch.bmm(hidden_states.transpose(0, 1), self.weight.transpose(1, 2)).transpose(0, 1).contiguous()


@auto_docstring(
    custom_intro="""
    The LFM3-Audio depth decoder with its per-codebook heads, predicting the masked codebooks of a speech frame from
    the revealed ones, as in [MaskGIT](https://huggingface.co/papers/2202.04200).
    """
)
class Lfm3AudioDepthDecoderForMaskedLM(Lfm3AudioPreTrainedModel):
    config: Lfm3AudioDepthDecoderConfig
    _tied_weights_keys = {"head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: Lfm3AudioDepthDecoderConfig):
        super().__init__(config)
        self.model = Lfm3AudioDepthDecoderModel(config)
        self.head = Lfm3AudioDepthDecoderHead(config)
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor,
        backbone_last_hidden_state: torch.FloatTensor,
        labels: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MaskedLMOutput:
        r"""
        input_ids (`torch.LongTensor` of shape `(num_frames, num_codebooks)`):
            Codes of each frame; `codebook_size` marks a masked codebook.
        backbone_last_hidden_state (`torch.FloatTensor` of shape `(num_frames, backbone_hidden_size)`):
            The language model's hidden state at the position preceding each frame.
        labels (`torch.LongTensor` of shape `(num_frames, num_codebooks)`, *optional*):
            Codes of the masked codebooks, `-100` elsewhere.
        """
        outputs = self.model(input_ids=input_ids, backbone_last_hidden_state=backbone_last_hidden_state, **kwargs)
        logits = self.head(outputs.last_hidden_state)

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.codebook_size,
                num_items_in_batch=(labels != -100).sum().clamp(min=1),
            )

        return MaskedLMOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )


@auto_docstring(
    custom_intro="""
    Base class for Lfm3Audio outputs, with hidden states and attentions.
    """
)
@dataclass
class Lfm3AudioModelOutputWithPast(BaseModelOutputWithPast):
    r"""
    audio_hidden_states (`torch.FloatTensor` of shape `(num_audio_tokens, hidden_size)`, *optional*):
        The projected speech encoder outputs scattered into the `audio_token_id` placeholders.
    padding_cache (`Lfm3AudioEncoderCausalConvPaddingCache`, *optional*):
        Subsampling state of the speech encoder, for the next chunk.
    encoder_cache (`Cache`, *optional*):
        Cache of the speech encoder, for the next chunk.
    """

    audio_hidden_states: torch.FloatTensor | None = None
    padding_cache: Lfm3AudioEncoderCausalConvPaddingCache | None = None
    encoder_cache: Cache | None = None


@auto_docstring(
    custom_intro="""
    Base class for Lfm3Audio causal language model (or autoregressive) outputs.
    """
)
@dataclass
class Lfm3AudioCausalLMOutputWithPast(CausalLMOutputWithPast):
    r"""
    audio_hidden_states (`torch.FloatTensor` of shape `(num_audio_tokens, hidden_size)`, *optional*):
        The projected speech encoder outputs scattered into the `audio_token_id` placeholders.
    text_loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        The language modeling loss of the text tokens.
    audio_loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `audio_labels` is provided):
        The depth decoder loss of the speech frames.
    last_hidden_state (`torch.FloatTensor` of shape `(batch_size, kept_length, hidden_size)`):
        The language model's hidden states behind `logits`, sliced like them.
    padding_cache (`Lfm3AudioEncoderCausalConvPaddingCache`, *optional*):
        Subsampling state of the speech encoder, for the next chunk.
    encoder_cache (`Cache`, *optional*):
        Cache of the speech encoder, for the next chunk.
    """

    audio_hidden_states: torch.FloatTensor | None = None
    text_loss: torch.FloatTensor | None = None
    audio_loss: torch.FloatTensor | None = None
    last_hidden_state: torch.FloatTensor | None = None
    padding_cache: Lfm3AudioEncoderCausalConvPaddingCache | None = None
    encoder_cache: Cache | None = None


class Lfm3AudioEmbeddings(HiggsAudioV2Embeddings):
    def __init__(self, config: Lfm3AudioConfig):
        super().__init__(config)
        self.embed_audio_tokens = nn.Embedding(
            config.num_codebooks * config.codebook_size, config.text_config.hidden_size
        )


@auto_docstring
class Lfm3AudioModel(Lfm3AudioPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [
        r"depth_embeddings\.\d+\.to_logits\.",
        r"\baudio_model\.",
        r"\bmulti_modal_projector\.",
        r"\bembed_audio_tokens\.",
        r"\bdepth_decoder\.",
    ]

    def __init__(self, config: Lfm3AudioConfig):
        super().__init__(config)
        self.language_model = AutoModel.from_config(config.text_config)
        self.audio_model = AutoModel.from_config(config.audio_config) if config.audio_config is not None else None
        self.multi_modal_projector = Lfm3AudioMultiModalProjector(config) if config.audio_config is not None else None
        self.embed_audio_tokens = Lfm3AudioEmbeddings(config) if config.depth_decoder_config is not None else None
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def get_audio_features(
        self,
        input_features: torch.FloatTensor,
        input_features_mask: torch.Tensor | None = None,
        padding_cache: Lfm3AudioEncoderCausalConvPaddingCache | None = None,
        encoder_cache: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Lfm3AudioAudioModelOutput:
        r"""
        input_features (`torch.FloatTensor` of shape `(num_clips, num_frames, num_mel_bins)`):
            Log-mel features from [`Lfm3AudioFeatureExtractor`], clips of all samples stacked and right-padded.
        input_features_mask (`torch.Tensor` of shape `(num_clips, num_frames)`, *optional*):
            Mask of the valid frames: 1 valid, 0 padding.
        padding_cache (`Lfm3AudioEncoderCausalConvPaddingCache`, *optional*):
            Subsampling state of the speech encoder, when streaming: the one returned for the previous chunk.
        encoder_cache (`Cache`, *optional*):
            Cache of the speech encoder, when streaming: the one returned for the previous chunk.

        Returns:
            [`Lfm3AudioAudioModelOutput`], `pooler_output`: projected valid tokens of all clips, flattened.
        """
        encoder_outputs = self.audio_model(
            input_features,
            input_features_mask=input_features_mask,
            padding_cache=padding_cache,
            cache=encoder_cache,
            use_cache=encoder_cache is not None,
            return_dict=True,
            **kwargs,
        )
        encoded = encoder_outputs.last_hidden_state
        if encoder_outputs.attention_mask is not None:
            encoded = encoded[encoder_outputs.attention_mask.bool().to(encoded.device)]

        encoder_outputs.pooler_output = self.multi_modal_projector(encoded.reshape(-1, encoded.shape[-1]))
        return encoder_outputs

    def get_audio_input_embeds(self, audio_input_ids: torch.LongTensor) -> torch.FloatTensor:
        r"""
        audio_input_ids (`torch.LongTensor` of shape `(num_frames, num_codebooks)`):
            The codes of the speech frames in the prompt.

        Returns:
            `torch.FloatTensor` of shape `(num_frames, hidden_size)`: the sum of the codebook embeddings of each frame.
        """
        codebook_size = self.config.codebook_size
        # an out-of-range code would read another codebook
        torch_compilable_check(
            (audio_input_ids >= 0) & (audio_input_ids < codebook_size),
            f"audio_input_ids must be in [0, {codebook_size})",
        )
        return self.embed_audio_tokens(audio_input_ids)

    def get_placeholder_mask(self, input_ids, inputs_embeds, token_id: int, num_features: int) -> torch.Tensor:
        if input_ids is not None:
            mask = input_ids == token_id
        else:
            token_embed = self.get_input_embeddings()(
                torch.full((), token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            mask = (inputs_embeds == token_embed).all(-1)

        torch_compilable_check(
            mask.sum() == num_features,
            lambda: f"Placeholder tokens and features do not match: {int(mask.sum())} tokens {token_id}, "
            f"{num_features} features",
        )
        return mask.unsqueeze(-1).to(inputs_embeds.device)

    @can_return_tuple
    @auto_docstring
    @deprecate_kwarg("past_key_values", "5.40.0", new_name="cache")
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        cache: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        input_features: torch.FloatTensor | None = None,
        input_features_mask: torch.Tensor | None = None,
        audio_input_ids: torch.LongTensor | None = None,
        padding_cache: Lfm3AudioEncoderCausalConvPaddingCache | None = None,
        encoder_cache: Cache | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Lfm3AudioModelOutputWithPast:
        r"""
        input_features (`torch.FloatTensor` of shape `(num_clips, num_frames, num_mel_bins)`, *optional*):
            Log-mel features from [`Lfm3AudioFeatureExtractor`], clips of all samples stacked and right-padded.
        input_features_mask (`torch.Tensor` of shape `(num_clips, num_frames)`, *optional*):
            Mask of the valid frames: 1 valid, 0 padding.
        audio_input_ids (`torch.LongTensor` of shape `(num_frames, num_codebooks)`, *optional*):
            The codes of the speech frames in the prompt, one row per `audio_out_token_id` placeholder.
        padding_cache (`Lfm3AudioEncoderCausalConvPaddingCache`, *optional*):
            Subsampling state of the speech encoder, when streaming: the one returned for the previous chunk.
        encoder_cache (`Cache`, *optional*):
            Cache of the speech encoder, when streaming: the one returned for the previous chunk.
        """
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You must specify input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        audio_embeds = None
        if input_features is not None:
            if self.audio_model is None:
                raise ValueError("`input_features` need the speech encoder, which `audio_config=None` dropped")

            audio_embeds = self.get_audio_features(
                input_features, input_features_mask, padding_cache, encoder_cache, return_dict=True
            ).pooler_output.to(inputs_embeds.device, inputs_embeds.dtype)
            audio_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds, self.config.audio_token_id, audio_embeds.shape[0]
            )
            inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_embeds)

        if audio_input_ids is not None:
            if self.embed_audio_tokens is None:
                raise ValueError(
                    "`audio_input_ids` need the speech frame embeddings, which `depth_decoder_config=None` dropped"
                )

            audio_out_embeds = self.get_audio_input_embeds(audio_input_ids).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            audio_out_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds, self.config.audio_out_token_id, audio_out_embeds.shape[0]
            )
            inputs_embeds = inputs_embeds.masked_scatter(audio_out_mask, audio_out_embeds)

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache=cache,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        return Lfm3AudioModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            cache=outputs.cache,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            audio_hidden_states=audio_embeds,
            padding_cache=padding_cache,
            encoder_cache=encoder_cache,
        )


@auto_docstring(
    custom_intro="""
    The LFM3-Audio model, generating text and speech.
    """
)
class Lfm3AudioForConditionalGeneration(Lfm3AudioPreTrainedModel, Lfm3AudioGenerationMixin):
    # Trainer counts text labels only; the joint loss also learns speech frames.
    accepts_loss_kwargs = False
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_gather_output"}

    def __init__(self, config: Lfm3AudioConfig):
        super().__init__(config)
        self.model = Lfm3AudioModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)

        self.depth_decoder = None
        if config.depth_decoder_config is not None:
            self.depth_decoder = Lfm3AudioDepthDecoderForMaskedLM._from_config(config.depth_decoder_config)
            self._tied_weights_keys = {
                **self._tied_weights_keys,
                "depth_decoder.head.weight": "depth_decoder.model.embed_tokens.weight",
            }

        self.post_init()

    @auto_docstring
    def get_audio_features(
        self,
        input_features: torch.FloatTensor,
        input_features_mask: torch.Tensor | None = None,
        padding_cache: Lfm3AudioEncoderCausalConvPaddingCache | None = None,
        encoder_cache: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Lfm3AudioAudioModelOutput:
        r"""
        input_features (`torch.FloatTensor` of shape `(num_clips, num_frames, num_mel_bins)`):
            Log-mel features from [`Lfm3AudioFeatureExtractor`], clips of all samples stacked and right-padded.
        input_features_mask (`torch.Tensor` of shape `(num_clips, num_frames)`, *optional*):
            Mask of the valid frames: 1 valid, 0 padding.
        padding_cache (`Lfm3AudioEncoderCausalConvPaddingCache`, *optional*):
            Subsampling state of the speech encoder, when streaming: the one returned for the previous chunk.
        encoder_cache (`Cache`, *optional*):
            Cache of the speech encoder, when streaming: the one returned for the previous chunk.
        """
        return self.model.get_audio_features(
            input_features, input_features_mask, padding_cache, encoder_cache, **kwargs
        )

    @can_return_tuple
    @auto_docstring
    @deprecate_kwarg("past_key_values", "5.40.0", new_name="cache")
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        cache: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        input_features: torch.FloatTensor | None = None,
        input_features_mask: torch.Tensor | None = None,
        audio_input_ids: torch.LongTensor | None = None,
        labels: torch.LongTensor | None = None,
        audio_labels: torch.LongTensor | None = None,
        audio_labels_mask: torch.BoolTensor | None = None,
        padding_cache: Lfm3AudioEncoderCausalConvPaddingCache | None = None,
        encoder_cache: Cache | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Lfm3AudioCausalLMOutputWithPast:
        r"""
        input_features (`torch.FloatTensor` of shape `(num_clips, num_frames, num_mel_bins)`, *optional*):
            Log-mel features from [`Lfm3AudioFeatureExtractor`], clips of all samples stacked and right-padded.
        input_features_mask (`torch.Tensor` of shape `(num_clips, num_frames)`, *optional*):
            Mask of the valid frames: 1 valid, 0 padding.
        audio_input_ids (`torch.LongTensor` of shape `(num_frames, num_codebooks)`, *optional*):
            The codes of the speech frames in the sequence, one row per `audio_out_token_id` placeholder.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            The text targets, `-100` for the positions not to learn, including the speech frames.
        audio_labels (`torch.LongTensor` of shape `(num_frames, num_codebooks)`, *optional*):
            Speech targets, one row per frame of `audio_input_ids`, `-100` for frames not to learn.
        audio_labels_mask (`torch.BoolTensor` of shape `(num_frames, num_codebooks)`, *optional*):
            Codebooks of each frame the depth decoder predicts, the others being given. Default: random MaskGIT mask
            (a random share per frame, at least one codebook, all of an end-of-audio frame).
        padding_cache (`Lfm3AudioEncoderCausalConvPaddingCache`, *optional*):
            Subsampling state of the speech encoder, when streaming: the one returned for the previous chunk.
        encoder_cache (`Cache`, *optional*):
            Cache of the speech encoder, when streaming: the one returned for the previous chunk.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache=cache,
            inputs_embeds=inputs_embeds,
            input_features=input_features,
            input_features_mask=input_features_mask,
            audio_input_ids=audio_input_ids,
            padding_cache=padding_cache,
            encoder_cache=encoder_cache,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = outputs[0]
        kept_hidden_states = slice_logits_to_keep(hidden_states, logits_to_keep)
        logits = self.lm_head(kept_hidden_states)

        loss = text_loss = audio_loss = None
        if labels is not None:
            # mean over learned tokens, zero without any
            text_count = (labels[..., 1:] != -100).sum()
            kwargs.setdefault("num_items_in_batch", text_count.clamp(min=1))
            text_loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
            )

        if audio_labels is not None:
            if self.depth_decoder is None:
                raise ValueError("`audio_labels` need the depth decoder, which `depth_decoder_config` disables")
            if input_ids is None or audio_input_ids is None:
                raise ValueError("`audio_labels` need `input_ids` and `audio_input_ids` to locate the speech frames")

            audio_loss, audio_count = self.audio_loss(
                input_ids, hidden_states, audio_input_ids, audio_labels, audio_labels_mask
            )

        if text_loss is not None and audio_loss is not None:
            # mean over learned text tokens and speech frames, weighted per modality
            device = text_loss.device
            text_weight = self.config.text_loss_weight * text_count.to(device)
            audio_weight = self.config.audio_loss_weight * audio_count.to(device)
            loss = (text_weight * text_loss + audio_weight * audio_loss.to(device)) / (
                text_weight + audio_weight + 1e-6
            )

        return Lfm3AudioCausalLMOutputWithPast(
            loss=loss if loss is not None else text_loss if text_loss is not None else audio_loss,
            text_loss=text_loss,
            audio_loss=audio_loss,
            logits=logits,
            cache=outputs.cache,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            audio_hidden_states=outputs.audio_hidden_states,
            last_hidden_state=kept_hidden_states,
            padding_cache=outputs.padding_cache,
            encoder_cache=outputs.encoder_cache,
        )

    def audio_loss(
        self,
        input_ids: torch.LongTensor,
        hidden_states: torch.FloatTensor,
        audio_input_ids: torch.LongTensor,
        audio_labels: torch.LongTensor,
        audio_labels_mask: torch.BoolTensor | None = None,
    ) -> tuple[torch.FloatTensor, torch.LongTensor]:
        """Depth decoder loss (mean over masked codes) and number of learned speech frames."""
        device = hidden_states.device
        audio_input_ids, audio_labels = audio_input_ids.to(device), audio_labels.to(device)
        is_learned = (audio_labels != -100).any(-1)

        # frame predicted from the hidden state before it
        batch_idx, frame_idx = (input_ids.to(device) == self.config.audio_out_token_id).nonzero(as_tuple=True)
        backbone_last_hidden_state = hidden_states[batch_idx[is_learned], frame_idx[is_learned] - 1]
        audio_input_ids, audio_labels = audio_input_ids[is_learned], audio_labels[is_learned]

        if audio_labels_mask is None:
            masked = self._maskgit_mask(audio_input_ids)
        else:
            masked = audio_labels_mask.to(device)[is_learned]

        loss = self.depth_decoder(
            input_ids=audio_input_ids.masked_fill(masked, self.config.codebook_size),
            backbone_last_hidden_state=backbone_last_hidden_state,
            labels=audio_labels.masked_fill(~masked, -100),
        ).loss

        return loss, is_learned.sum()

    def _maskgit_mask(self, codes: torch.LongTensor) -> torch.BoolTensor:
        """MaskGIT mask: random share of each frame's codebooks, at least one; all of an end-of-audio frame."""
        ratio = torch.rand(codes.shape[0], 1, device=codes.device)
        masked = torch.rand(codes.shape, device=codes.device) < ratio
        masked |= ~masked.any(-1, keepdim=True)
        masked |= (codes == self.config.audio_eos_code).all(-1, keepdim=True)
        return masked


__all__ = [
    "Lfm3AudioConfig",
    "Lfm3AudioAudioConfig",
    "Lfm3AudioAudioModel",
    "Lfm3AudioDepthDecoderConfig",
    "Lfm3AudioDepthDecoderModel",
    "Lfm3AudioDepthDecoderForMaskedLM",
    "Lfm3AudioPreTrainedModel",
    "Lfm3AudioModel",
    "Lfm3AudioForConditionalGeneration",
]
