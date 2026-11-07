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
"""LFM3-Audio: LFM3 language model with a Liquid speech encoder and codebook speech output."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict
from torch import nn

from ... import initialization as init
from ...cache_utils import Cache, DynamicCache
from ...configuration_utils import PreTrainedConfig, SubConfigSpec
from ...generation import GenerationConfig
from ...modeling_outputs import BaseModelOutputWithPast, BaseModelOutputWithPooling, CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel, slice_logits_to_keep
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, torch_compilable_check
from ...utils.import_utils import is_tracing
from ..auto import AutoConfig, AutoModel
from ..lfm3.modeling_lfm3 import Lfm3PreTrainedModel
from .generation_lfm3_audio import Lfm3AudioGenerationMixin


@auto_docstring(checkpoint="LiquidAI/LFM3-Audio-2B")
@strict
class Lfm3AudioEncoderConfig(PreTrainedConfig):
    r"""
    text_config (`Lfm3Config` or `dict`, *optional*):
        The configuration of the LFM3 model run over the downsampled spectrogram: sliding-window attention over
        `sliding_window` frames, no attention gate, no sandwich norm. Defaults to the 17-layer, 768-wide model of
        LFM3-Audio.
    conv_channels (`int`, *optional*, defaults to 256):
        The number of channels of the downsampling convolutions.
    """

    model_type = "lfm3_audio_encoder"
    sub_configs_defaults = {
        "text_config": SubConfigSpec(
            config_class=AutoConfig,
            model_type="lfm3",
            init_kwargs={
                "vocab_size": 2,
                "hidden_size": 768,
                "intermediate_size": 1792,
                "num_hidden_layers": 17,
                "num_attention_heads": 8,
                "num_key_value_heads": 4,
                "head_dim": 96,
                "layer_types": ["conv", "conv", "sliding_attention"] * 5 + ["sliding_attention"] * 2,
                "sliding_window": 32,
                "conv_L_cache": 3,
                "rope_parameters": {"sliding_attention": {"rope_type": "default", "rope_theta": 1_000_000.0}},
                "use_sandwich_norm": False,
                "use_attention_gate": False,
                "pad_token_id": 0,
                "bos_token_id": 1,
                "eos_token_id": None,
            },
        ),
    }

    text_config: dict | PreTrainedConfig | None = None
    num_mel_bins: int = 128
    conv_channels: int = 256
    initializer_range: float = 0.02


@auto_docstring(checkpoint="LiquidAI/LFM3-Audio-2B")
@strict
class Lfm3AudioConfig(PreTrainedConfig):
    r"""
    audio_encoder_config (`Lfm3AudioEncoderConfig` or `dict`, *optional*):
        The configuration of the speech encoder.
    depth_config (`Lfm3Config` or `dict`, *optional*):
        The configuration of the depth decoder, a bidirectional LFM3 model over the codebook axis of one speech
        frame. Defaults to the 3-layer, 1024-wide model of LFM3-Audio.
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
    text_loss_weight (`float`, *optional*, defaults to 1.0):
        The weight of the text loss in the training loss.
    audio_loss_weight (`float`, *optional*, defaults to 1.0):
        The weight of the speech loss in the training loss.
    """

    model_type = "lfm3_audio"
    sub_configs_defaults = {
        "text_config": SubConfigSpec(config_class=AutoConfig, model_type="lfm3"),
        "audio_encoder_config": SubConfigSpec(config_class=Lfm3AudioEncoderConfig),
        "depth_config": SubConfigSpec(
            config_class=AutoConfig,
            model_type="lfm3",
            init_kwargs={
                "vocab_size": 1,
                "pad_token_id": None,
                "bos_token_id": None,
                "eos_token_id": None,
                "hidden_size": 1024,
                "intermediate_size": 2816,
                "num_hidden_layers": 3,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "head_dim": 32,
                "max_position_embeddings": 12,
                "layer_types": ["full_attention"] * 3,
                "rope_parameters": {
                    "full_attention": {"rope_type": "default", "rope_theta": 1_000_000.0, "partial_rotary_factor": 1.0}
                },
                "sliding_window": None,
                "use_sandwich_norm": False,
                "use_cache": False,
            },
        ),
    }

    text_config: dict | PreTrainedConfig | None = None
    audio_encoder_config: dict | PreTrainedConfig | None = None
    depth_config: dict | PreTrainedConfig | None = None
    num_codebooks: int = 12
    codebook_size: int = 2017
    audio_eos_code: int = 2016
    audio_token_id: int = 125_031
    audio_out_token_id: int = 125_032
    audio_start_token_id: int = 125_012
    text_end_token_id: int = 125_014
    mixed_start_token_id: int = 125_015
    tie_word_embeddings: bool = True
    text_loss_weight: float = 1.0
    audio_loss_weight: float = 1.0
    initializer_range: float = 0.02

    def __post_init__(self, **kwargs):
        if isinstance(self.depth_config, dict):
            # the depth decoder consumes embeddings only; its token ids are placeholders
            self.depth_config = {"pad_token_id": None, "bos_token_id": None, "eos_token_id": None, **self.depth_config}
        super().__post_init__(**kwargs)

    @property
    def audio_config(self):
        return self.audio_encoder_config


class Lfm3AudioDownsampler(nn.Module):
    """Three stride-2 convolutions over the (mel, time) plane: 128 mel bins and 8 frames become one vector."""

    def __init__(self, config: Lfm3AudioEncoderConfig):
        super().__init__()
        channels = config.conv_channels
        conv = {"kernel_size": 3, "stride": 2, "padding": 1}
        self.conv_in = nn.Conv2d(1, channels, **conv)
        self.conv_mid = nn.Conv2d(channels, channels, groups=channels, **conv)
        self.lin_mid = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv_end = nn.Conv2d(channels, channels, groups=channels, **conv)
        self.lin_end = nn.Conv2d(channels, channels, kernel_size=1)
        self.lin_out = nn.Conv1d(config.num_mel_bins * channels // 8, config.text_config.hidden_size, kernel_size=1)
        self.act = nn.SiLU()

    def forward(self, input_features: torch.Tensor, feature_lengths: torch.Tensor | None = None) -> torch.Tensor:
        def mask_padding(hidden_states, lengths):
            if lengths is None:
                return hidden_states
            valid = torch.arange(hidden_states.shape[-1], device=hidden_states.device)[None] < lengths[:, None]
            return hidden_states * valid[:, None, None].to(hidden_states.dtype)

        hidden_states = mask_padding(input_features.unsqueeze(1), feature_lengths)
        hidden_states = self.act(self.conv_in(hidden_states))
        lengths = None if feature_lengths is None else (feature_lengths + 1) // 2
        hidden_states = mask_padding(hidden_states, lengths)
        hidden_states = self.act(self.lin_mid(self.conv_mid(hidden_states)))
        lengths = None if lengths is None else (lengths + 1) // 2
        hidden_states = mask_padding(hidden_states, lengths)
        hidden_states = self.act(self.lin_end(self.conv_end(hidden_states)))
        return self.lin_out(hidden_states.flatten(1, 2))

    @staticmethod
    def output_lengths(lengths: torch.Tensor) -> torch.Tensor:
        return -(lengths // -8)


class Lfm3AudioEncoderStreamingCache:
    """Bounded downsampler context and causal encoder state for synchronized audio chunks."""

    def __init__(self, config: Lfm3AudioEncoderConfig):
        self.past_key_values = DynamicCache(config=config.text_config)
        self.features: torch.Tensor | None = None
        self.offset = 0
        self.num_frames = 0
        self.num_tokens = 0


class Lfm3AudioEncoder(nn.Module):
    def __init__(self, config: Lfm3AudioEncoderConfig):
        super().__init__()
        self.config = config
        self.downsample = Lfm3AudioDownsampler(config)
        self.lfm = AutoModel.from_config(config.text_config)
        # the encoder only sees `inputs_embeds`
        self.lfm.register_module("embed_tokens", None)

    def forward(
        self, input_features: torch.Tensor, feature_lengths: torch.Tensor, **kwargs: Unpack[TransformersKwargs]
    ) -> tuple[BaseModelOutputWithPast, torch.Tensor]:
        hidden_states = self.downsample(input_features, feature_lengths).transpose(1, 2)
        # the encoder config is the one a caller sets the output controls on
        for key in ("output_attentions", "output_hidden_states"):
            if getattr(self.config, key):
                kwargs.setdefault(key, True)
        outputs = self.lfm(inputs_embeds=hidden_states, use_cache=False, return_dict=True, **kwargs)
        return outputs, self.downsample.output_lengths(feature_lengths)

    @torch.no_grad()
    def stream(
        self,
        input_features: torch.Tensor,
        cache: Lfm3AudioEncoderStreamingCache | None = None,
        is_final: bool = False,
    ) -> tuple[torch.Tensor, Lfm3AudioEncoderStreamingCache]:
        """Encode complete stride-8 windows, with seven frames of lookahead and bounded left context."""
        if cache is None:
            cache = Lfm3AudioEncoderStreamingCache(self.config)
        features = input_features.to(self.downsample.conv_in.weight.dtype)
        if cache.features is not None:
            features = torch.cat((cache.features, features), dim=-1)
        num_frames = cache.num_frames + input_features.shape[-1]
        num_tokens = (num_frames + (7 if is_final else 0)) // 8
        count = num_tokens - cache.num_tokens
        if count:
            start = cache.num_tokens - cache.offset // 8
            hidden_states = self.downsample(features).transpose(1, 2)[:, start : start + count]
            positions = torch.arange(cache.num_tokens, num_tokens, device=features.device)[None]
            encoded = self.lfm(
                inputs_embeds=hidden_states,
                past_key_values=cache.past_key_values,
                position_ids=positions,
                use_cache=True,
            ).last_hidden_state
        else:
            encoded = features.new_empty((features.shape[0], 0, self.config.text_config.hidden_size))
        offset = max(0, (num_tokens - 1) * 8)
        cache.features = features[..., offset - cache.offset :].clone() if not is_final else None
        cache.num_frames, cache.offset, cache.num_tokens = num_frames, offset, num_tokens
        return encoded, cache


class Lfm3AudioProjector(nn.Module):
    def __init__(self, config: Lfm3AudioConfig):
        super().__init__()
        encoder_config = config.audio_encoder_config
        in_features = encoder_config.text_config.hidden_size
        out_features = config.text_config.hidden_size
        self.layer_norm = nn.LayerNorm(in_features)
        self.linear_1 = nn.Linear(in_features, out_features)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(out_features, out_features)

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.act(self.linear_1(self.layer_norm(audio_features))))


class Lfm3AudioCodebookNorm(nn.Module):
    """One RMSNorm per codebook over the depth decoder's output slots, applied in float32 like `Olmo2RMSNorm`."""

    def __init__(self, config: Lfm3AudioConfig):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(config.num_codebooks, config.depth_config.hidden_size))
        self.variance_epsilon = config.depth_config.norm_eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (hidden_states * self.weight).to(input_dtype)


class Lfm3AudioDepthDecoder(nn.Module):
    """
    Predicts the codebooks of one speech frame: a bidirectional LFM3 over one slot per codebook, each holding its
    code embedding or a mask embedding, scored against the tied codebook embeddings.
    """

    def __init__(self, config: Lfm3AudioConfig):
        super().__init__()
        self.config = config
        depth_config = config.depth_config
        self.num_codebooks = config.num_codebooks
        self.codebook_size = config.codebook_size
        self.hidden_size = depth_config.hidden_size
        self.depth_linear = nn.Linear(config.text_config.hidden_size, self.num_codebooks * self.hidden_size)
        self.mask_embedding = nn.Parameter(torch.empty(self.hidden_size))
        self.model = AutoModel.from_config(depth_config)
        self.model.register_module("embed_tokens", None)
        self.codebook_embedding = nn.Parameter(torch.empty(self.num_codebooks, self.codebook_size, self.hidden_size))
        self.codebook_norm = Lfm3AudioCodebookNorm(config)
        # `repetition_window` is the number of past frames the repetition penalty covers
        self.generation_config = GenerationConfig(do_sample=True, repetition_window=64)

    def embed_codes(self, codes: torch.Tensor) -> torch.Tensor:
        offsets = torch.arange(self.num_codebooks, device=codes.device) * self.codebook_size
        return F.embedding(codes + offsets, self.codebook_embedding.flatten(0, 1))

    def forward(self, hidden_states: torch.Tensor, codes: torch.Tensor, masked: torch.Tensor) -> torch.Tensor:
        r"""
        hidden_states (`torch.FloatTensor` of shape `(num_frames, hidden_size)`):
            The language model's hidden state at the position preceding each frame.
        codes (`torch.LongTensor` of shape `(num_frames, num_codebooks)`):
            The codes of each frame.
        masked (`torch.BoolTensor` of shape `(num_frames, num_codebooks)`):
            The codebooks hidden from the decoder, which it is asked to predict.

        Returns:
            `torch.FloatTensor` of shape `(num_frames, num_codebooks, codebook_size)`: the logits of every codebook.
        """
        depth_in = self.depth_linear(hidden_states).unflatten(-1, (self.num_codebooks, self.hidden_size))
        code_embeds = self.embed_codes(codes)
        depth_in = depth_in + torch.where(masked[..., None], self.mask_embedding.to(code_embeds.dtype), code_embeds)
        depth_out = self.codebook_norm(self.trunk(depth_in))
        codebook_embedding = self.codebook_embedding.to(depth_out.dtype)
        logits = torch.matmul(depth_out.transpose(0, 1), codebook_embedding.transpose(1, 2))
        return logits.transpose(0, 1)

    def trunk(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # the codebooks of a frame are predicted jointly: bidirectional attention
        return self.model(
            inputs_embeds=hidden_states,
            attention_mask={"full_attention": None},
            is_causal=False,
            use_cache=False,
        ).last_hidden_state


@auto_docstring
class Lfm3AudioPreTrainedModel(Lfm3PreTrainedModel):
    config: Lfm3AudioConfig
    base_model_prefix = "model"
    input_modalities = ("audio", "text")
    _no_split_modules = ["Lfm3AudioDepthDecoder"]
    _can_record_outputs = AttributeError()

    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, Lfm3AudioDepthDecoder):
            init.normal_(module.mask_embedding, std=0.02)
            init.normal_(module.codebook_embedding, std=self.config.initializer_range)
        elif isinstance(module, Lfm3AudioCodebookNorm):
            init.ones_(module.weight)


@auto_docstring(
    custom_intro="""
    Base class for Lfm3Audio outputs, with hidden states and attentions.
    """
)
@dataclass
class Lfm3AudioModelOutputWithPast(BaseModelOutputWithPast):
    r"""
    audio_hidden_states (`torch.FloatTensor`, *optional*):
        A `torch.FloatTensor` of size `(num_audio_tokens, hidden_size)`.
        audio_hidden_states of the model produced by the speech encoder and after projecting the last hidden state.
    """

    audio_hidden_states: torch.FloatTensor | None = None


@auto_docstring(
    custom_intro="""
    Base class for Lfm3Audio causal language model (or autoregressive) outputs.
    """
)
@dataclass
class Lfm3AudioCausalLMOutputWithPast(CausalLMOutputWithPast):
    r"""
    audio_hidden_states (`torch.FloatTensor`, *optional*):
        A `torch.FloatTensor` of size `(num_audio_tokens, hidden_size)`.
        audio_hidden_states of the model produced by the speech encoder and after projecting the last hidden state.
    text_loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        The language modeling loss of the text tokens.
    last_hidden_state (`torch.FloatTensor` of shape `(batch_size, kept_length, hidden_size)`):
        The language model's hidden states behind `logits`, which the depth decoder reads to sample speech frames.
    audio_loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `audio_labels` is provided):
        The depth decoder loss of the speech frames.
    """

    audio_hidden_states: torch.FloatTensor | None = None
    last_hidden_state: torch.FloatTensor | None = None
    text_loss: torch.FloatTensor | None = None
    audio_loss: torch.FloatTensor | None = None


@auto_docstring
class Lfm3AudioModel(Lfm3AudioPreTrainedModel):
    def __init__(self, config: Lfm3AudioConfig):
        super().__init__(config)
        self.language_model = AutoModel.from_config(config.text_config)
        self.audio_encoder = Lfm3AudioEncoder(config.audio_encoder_config)
        self.audio_projector = Lfm3AudioProjector(config)
        self.codebook_embedding = nn.Embedding(
            config.num_codebooks * config.codebook_size, config.text_config.hidden_size
        )
        self.depth_decoder = Lfm3AudioDepthDecoder(config)
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    @can_return_tuple
    @auto_docstring
    def get_audio_features(
        self,
        input_features: torch.FloatTensor,
        feature_attention_mask: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        input_features (`torch.FloatTensor` of shape `(num_clips, num_mel_bins, num_frames)`):
            The log-mel spectrograms of the input clips, right-padded to the longest clip.
        feature_attention_mask (`torch.Tensor` of shape `(num_clips, num_frames)`):
            The mask of the valid spectrogram frames of each clip.
        """
        feature_lengths = feature_attention_mask.sum(-1)
        input_features = input_features.to(self.audio_encoder.downsample.conv_in.weight.dtype)
        encoder_outputs, encoded_lengths = self.audio_encoder(input_features, feature_lengths, **kwargs)
        encoded = encoder_outputs.last_hidden_state
        valid = torch.arange(encoded.shape[1], device=encoded.device)[None, :] < encoded_lengths[:, None].to(
            encoded.device
        )
        # `pooler_output`: the projected valid tokens of all clips, flattened in order
        return BaseModelOutputWithPooling(
            last_hidden_state=encoded,
            pooler_output=self.audio_projector(encoded[valid]),
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    @torch.no_grad()
    def get_audio_features_stream(
        self,
        input_features: torch.FloatTensor,
        cache: Lfm3AudioEncoderStreamingCache | None = None,
        is_final: bool = False,
    ) -> tuple[torch.Tensor, Lfm3AudioEncoderStreamingCache]:
        """Project newly completed audio tokens, returning `(batch, tokens, hidden)` and the updated cache."""
        encoded, cache = self.audio_encoder.stream(input_features, cache, is_final)
        return self.audio_projector(encoded), cache

    def get_audio_out_embeds(self, audio_out_codes: torch.LongTensor) -> torch.FloatTensor:
        r"""
        audio_out_codes (`torch.LongTensor` of shape `(num_frames, num_codebooks)`):
            The codes of the speech frames in the prompt.

        Returns:
            `torch.FloatTensor` of shape `(num_frames, hidden_size)`: the sum of the codebook embeddings of each frame.
        """
        num_codebooks, codebook_size = self.config.num_codebooks, self.config.codebook_size
        # data-dependent, so a traced or captured forward skips it; an out-of-range code would read another codebook
        if not is_tracing(audio_out_codes):
            torch_compilable_check(
                (audio_out_codes >= 0) & (audio_out_codes < codebook_size),
                f"audio_out_codes must be in [0, {codebook_size})",
            )
        offsets = torch.arange(num_codebooks, device=audio_out_codes.device) * codebook_size
        return self.codebook_embedding(audio_out_codes + offsets).sum(-2)

    def get_placeholder_mask(self, input_ids, inputs_embeds, token_id: int, num_features: int) -> torch.Tensor:
        if input_ids is not None:
            mask = input_ids == token_id
        else:
            token_embed = self.get_input_embeddings()(
                torch.full((), token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            mask = (inputs_embeds == token_embed).all(-1)
        if not is_tracing(mask):
            torch_compilable_check(
                mask.sum() == num_features,
                lambda: f"Placeholder tokens and features do not match: {int(mask.sum())} tokens {token_id}, "
                f"{num_features} features",
            )
        return mask.unsqueeze(-1).to(inputs_embeds.device)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        input_features: torch.FloatTensor | None = None,
        feature_attention_mask: torch.Tensor | None = None,
        audio_out_codes: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Lfm3AudioModelOutputWithPast:
        r"""
        input_features (`torch.FloatTensor` of shape `(num_clips, num_mel_bins, num_frames)`, *optional*):
            The log-mel spectrograms of the input clips, right-padded to the longest clip.
        feature_attention_mask (`torch.Tensor` of shape `(num_clips, num_frames)`, *optional*):
            The mask of the valid spectrogram frames of each clip.
        audio_out_codes (`torch.LongTensor` of shape `(num_frames, num_codebooks)`, *optional*):
            The codes of the speech frames in the prompt, one row per `audio_out_token_id` placeholder.
        """
        # both may come: `inputs_embeds` with features merged, `input_ids` for the placeholders
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You must specify input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        audio_features = None
        if input_features is not None:
            if feature_attention_mask is None:
                feature_attention_mask = torch.ones(
                    input_features.shape[0], input_features.shape[-1], dtype=torch.bool, device=input_features.device
                )
            audio_features = self.get_audio_features(
                input_features, feature_attention_mask, return_dict=True
            ).pooler_output.to(inputs_embeds.device, inputs_embeds.dtype)
            audio_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds, self.config.audio_token_id, audio_features.shape[0]
            )
            inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_features)
        if audio_out_codes is not None:
            audio_out_embeds = self.get_audio_out_embeds(audio_out_codes).to(inputs_embeds.device, inputs_embeds.dtype)
            audio_out_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds, self.config.audio_out_token_id, audio_out_embeds.shape[0]
            )
            inputs_embeds = inputs_embeds.masked_scatter(audio_out_mask, audio_out_embeds)

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )
        return Lfm3AudioModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            audio_hidden_states=audio_features,
        )


@auto_docstring(
    custom_intro="""
    The LFM3-Audio model: an LFM3 language model that reads speech through the Liquid speech encoder and writes
    speech as codebook frames through a depth decoder, alongside text.
    """
)
class Lfm3AudioForConditionalGeneration(Lfm3AudioPreTrainedModel, Lfm3AudioGenerationMixin):
    # Trainer counts text labels only; the joint loss also supervises speech frames.
    accepts_loss_kwargs = False
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}

    def __init__(self, config: Lfm3AudioConfig):
        super().__init__(config)
        self.model = Lfm3AudioModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    @auto_docstring
    def get_audio_features(
        self,
        input_features: torch.FloatTensor,
        feature_attention_mask: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        input_features (`torch.FloatTensor` of shape `(num_clips, num_mel_bins, num_frames)`):
            The log-mel spectrograms of the input clips, right-padded to the longest clip.
        feature_attention_mask (`torch.Tensor` of shape `(num_clips, num_frames)`):
            The mask of the valid spectrogram frames of each clip.
        """
        return self.model.get_audio_features(input_features, feature_attention_mask, **kwargs)

    @property
    def language_model(self):
        return self.model.language_model

    @property
    def _placeholder_token_ids(self) -> list[int]:
        # never sampled as text: one in `sequences` would break the one-placeholder-per-frame contract
        return [self.config.audio_token_id, self.config.audio_out_token_id]

    @property
    def depth_decoder(self):
        return self.model.depth_decoder

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        loaded = super().from_pretrained(*args, **kwargs)
        model = loaded[0] if kwargs.get("output_loading_info", False) else loaded
        # the depth decoder's generation config is serialized under the `depth_decoder_` prefix of the model's
        prefix = "depth_decoder_"
        depth_decoder_attrs = {
            attr[len(prefix) :]: value
            for attr, value in vars(model.generation_config).items()
            if attr.startswith(prefix)
        }
        vars(model.depth_decoder.generation_config).update({"_from_model_config": False, **depth_decoder_attrs})
        for attr in depth_decoder_attrs:
            delattr(model.generation_config, prefix + attr)
        return loaded

    def save_pretrained(self, *args, **kwargs):
        prefix = "depth_decoder_"
        # rebuilt from the live depth settings, so a setting reset to its default disappears
        for attr in list(vars(self.generation_config)):
            if attr.startswith(prefix):
                delattr(self.generation_config, attr)
        depth_decoder_attrs = self.depth_decoder.generation_config.to_diff_dict()
        for key in ("transformers_version", "_from_model_config"):
            depth_decoder_attrs.pop(key, None)
        for attr, value in depth_decoder_attrs.items():
            setattr(self.generation_config, prefix + attr, value)
        # a generation config flagged as derived from the model config drops custom attributes when it is read back
        self.generation_config._from_model_config = False
        try:
            super().save_pretrained(*args, **kwargs)
        finally:
            for attr in list(vars(self.generation_config)):
                if attr.startswith(prefix):
                    delattr(self.generation_config, attr)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        input_features: torch.FloatTensor | None = None,
        feature_attention_mask: torch.Tensor | None = None,
        audio_out_codes: torch.LongTensor | None = None,
        labels: torch.LongTensor | None = None,
        audio_labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Lfm3AudioCausalLMOutputWithPast:
        r"""
        input_features (`torch.FloatTensor` of shape `(num_clips, num_mel_bins, num_frames)`, *optional*):
            The log-mel spectrograms of the input clips, right-padded to the longest clip.
        feature_attention_mask (`torch.Tensor` of shape `(num_clips, num_frames)`, *optional*):
            The mask of the valid spectrogram frames of each clip.
        audio_out_codes (`torch.LongTensor` of shape `(num_frames, num_codebooks)`, *optional*):
            The codes of the speech frames in the sequence, one row per `audio_out_token_id` placeholder.
        audio_labels (`torch.LongTensor` of shape `(num_frames, num_codebooks)`, *optional*):
            The codes of the speech frames to learn, one row per `audio_out_token_id` placeholder; rows of `-100` are
            ignored. The depth decoder predicts a random subset of each frame's codebooks from the others and the
            hidden state preceding the frame.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            audio_out_codes=audio_out_codes,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = outputs[0]
        kept_hidden_states = slice_logits_to_keep(hidden_states, logits_to_keep)
        logits = self.lm_head(kept_hidden_states)

        loss = text_loss = audio_loss = None
        audio_count = 0
        if labels is not None:
            text_loss = (
                self.loss_function(
                    logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
                )
                if (labels[..., 1:] != -100).any()
                else logits.sum() * 0.0
            )
        if audio_labels is not None:
            if input_ids is None:
                raise ValueError("`audio_labels` requires `input_ids` to locate the speech frames")
            audio_loss, audio_count = self.audio_loss(input_ids, hidden_states, audio_labels)
        if text_loss is not None or audio_loss is not None:
            loss = self.combined_loss(labels, audio_count, text_loss, audio_loss)

        return Lfm3AudioCausalLMOutputWithPast(
            loss=loss,
            text_loss=text_loss,
            audio_loss=audio_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            audio_hidden_states=outputs.audio_hidden_states,
            last_hidden_state=kept_hidden_states,
        )

    def audio_loss(
        self, input_ids: torch.LongTensor, hidden_states: torch.FloatTensor, audio_labels: torch.LongTensor
    ) -> tuple[torch.FloatTensor, torch.LongTensor]:
        """The depth decoder loss and the number of speech frames it supervises."""
        config = self.config
        # under `device_map` the hidden states sit on the last layer's device
        input_ids, audio_labels = input_ids.to(hidden_states.device), audio_labels.to(hidden_states.device)
        placeholders = input_ids == config.audio_out_token_id
        # a frame is predicted from the position before it: the first position of a row has none
        preceding = placeholders.clone()
        preceding[:, 0] = False
        supervised = (audio_labels != -100).all(-1)[preceding[placeholders]]
        latents = hidden_states[preceding.roll(-1, dims=1)][supervised]
        codes = audio_labels[preceding[placeholders]][supervised]
        if codes.shape[0] == 0:
            return hidden_states.sum() * 0.0, supervised.sum()
        ratio = torch.rand(codes.shape[0], 1, device=codes.device)
        masked = torch.rand(codes.shape, device=codes.device) < ratio
        masked |= ~masked.any(-1, keepdim=True)
        masked |= (codes == config.audio_eos_code).all(-1, keepdim=True)
        logits = self.model.depth_decoder(latents, codes, masked)
        # under `device_map` the depth decoder answers on its own device
        codes, masked = codes.to(logits.device), masked.to(logits.device)
        losses = F.cross_entropy(logits.transpose(1, 2).float(), codes, reduction="none")
        return (losses * masked).sum() / masked.sum().clamp_min(1), supervised.sum()

    def combined_loss(self, labels, audio_count, text_loss, audio_loss) -> torch.Tensor:
        """Mean over the supervised text tokens and speech frames, each modality weighted by count and config."""
        # under `device_map` the labels and the two losses can sit on three devices
        device = (text_loss if text_loss is not None else audio_loss).device
        text_count = 0 if labels is None else (labels[..., 1:] != -100).sum().to(device)
        if audio_loss is not None:
            audio_count = audio_count.to(device)
        text_weight = self.config.text_loss_weight * text_count
        audio_weight = self.config.audio_loss_weight * audio_count
        return (
            text_weight * (text_loss if text_loss is not None else 0.0)
            + audio_weight * (audio_loss.to(device) if audio_loss is not None else 0.0)
        ) / (text_weight + audio_weight + 1e-6)


__all__ = [
    "Lfm3AudioConfig",
    "Lfm3AudioEncoderConfig",
    "Lfm3AudioEncoderStreamingCache",
    "Lfm3AudioPreTrainedModel",
    "Lfm3AudioModel",
    "Lfm3AudioForConditionalGeneration",
]
